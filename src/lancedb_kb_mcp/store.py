"""A knowledge base in one LanceDB table: notes and file chunks, with vectors,
full-text search, metadata filters and versions.

Everything lives under one directory (LANCE_KB_DIR, default ./kb-data), so the
whole knowledge base can be copied, zipped, or put on S3 (lancedb.connect also
accepts s3:// URIs).
"""

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from .chunking import chunk_text
from .embed import get_embedder

TABLE = "items"
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def _score(hit):
    """Higher is better in every mode: RRF relevance, BM25 score, or cosine similarity."""
    if "_relevance_score" in hit:
        return hit["_relevance_score"]
    if "_distance" in hit:
        return 1 - hit["_distance"]
    return hit.get("_score")


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sql(value):
    return "'" + str(value).replace("'", "''") + "'"


class KnowledgeBase:
    def __init__(self, uri=None, embedder=None):
        self.uri = str(uri or os.environ.get("LANCE_KB_DIR", "kb-data"))
        self.db = lancedb.connect(self.uri)
        self.embedder = embedder or get_embedder()
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("kind", pa.string()),  # "note" or "chunk"
                pa.field("title", pa.string()),
                pa.field("text", pa.string()),
                pa.field("project", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("source", pa.string()),  # file path for chunks, "" for notes
                pa.field("line_start", pa.int32()),
                pa.field("line_end", pa.int32()),
                pa.field("content_hash", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedder.dim)),
            ]
        )
        if TABLE in self.db.list_tables().tables:
            self.table = self.db.open_table(TABLE)
        else:
            self.table = self.db.create_table(TABLE, schema=self.schema)

    # -- writing -----------------------------------------------------------
    def _rows(self, items):
        vectors = self.embedder.encode([f"{i['title']}\n{i['text']}" for i in items])
        return [{**i, "vector": v} for i, v in zip(items, vectors)]

    def _reindex(self):
        if self.table.count_rows():
            self.table.create_index("text", config=FTS(), replace=True)

    def add_note(self, title, text, project="default", tags=()):
        now = _now()
        row = {
            "id": uuid.uuid4().hex[:12],
            "kind": "note",
            "title": title,
            "text": text,
            "project": project,
            "tags": list(tags),
            "source": "",
            "line_start": 0,
            "line_end": 0,
            "content_hash": hashlib.sha1(text.encode()).hexdigest(),
            "created_at": now,
            "updated_at": now,
        }
        self.table.add(self._rows([row]))
        self._reindex()
        return row["id"]

    def update_note(self, note_id, title=None, text=None, project=None, tags=None):
        current = self.get(note_id)
        if current is None or current["kind"] != "note":
            raise KeyError(f"no note with id {note_id}")
        row = {
            k: v for k, v in current.items() if k in self.schema.names and k != "vector"
        }
        for key, value in (
            ("title", title),
            ("text", text),
            ("project", project),
            ("tags", tags),
        ):
            if value is not None:
                row[key] = list(value) if key == "tags" else value
        row["content_hash"] = hashlib.sha1(row["text"].encode()).hexdigest()
        row["updated_at"] = _now()
        self.table.merge_insert("id").when_matched_update_all().execute(
            self._rows([row])
        )
        self._reindex()
        return note_id

    def delete(self, item_id):
        self.table.delete(f"id = {_sql(item_id)}")
        self._reindex()

    def ingest_folder(self, folder, project="default", suffixes=TEXT_SUFFIXES):
        """Index text files under folder. Unchanged files are skipped; a changed
        file's chunks are replaced. Returns counts of files indexed and skipped."""
        folder = Path(folder).resolve()
        indexed = skipped = 0
        for path in sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            digest = hashlib.sha1(text.encode()).hexdigest()
            # Relative to the ingested folder, so a shared KB does not carry anyone's home path.
            source = f"{folder.name}/{path.relative_to(folder).as_posix()}"
            existing = (
                self.table.search()
                .where(f"source = {_sql(source)}")
                .select(["content_hash"])
                .limit(1)
                .to_list()
            )
            if existing and existing[0]["content_hash"] == digest:
                skipped += 1
                continue
            self.table.delete(f"source = {_sql(source)}")
            now = _now()
            rows = [
                {
                    "id": uuid.uuid4().hex[:12],
                    "kind": "chunk",
                    "title": heading or path.name,
                    "text": chunk,
                    "project": project,
                    "tags": [],
                    "source": source,
                    "line_start": start,
                    "line_end": end,
                    "content_hash": digest,
                    "created_at": now,
                    "updated_at": now,
                }
                for heading, chunk, start, end in chunk_text(text)
            ]
            if rows:
                self.table.add(self._rows(rows))
            indexed += 1
        self._reindex()
        return {"indexed": indexed, "skipped": skipped}

    # -- reading -----------------------------------------------------------
    def get(self, item_id):
        rows = self.table.search().where(f"id = {_sql(item_id)}").limit(1).to_list()
        return rows[0] if rows else None

    def search(
        self, query, k=8, project=None, kind=None, tag=None, version=None, mode="hybrid"
    ):
        """Hybrid (vector + full-text, merged with RRF), "vector" or "text" search."""
        table = self.table
        if version is not None:
            table = self.db.open_table(TABLE)
            table.checkout(version)
        filters = []
        if project:
            filters.append(f"project = {_sql(project)}")
        if kind:
            filters.append(f"kind = {_sql(kind)}")
        if tag:
            filters.append(f"array_has(tags, {_sql(tag)})")
        vector = self.embedder.encode([query], query=True)[0]
        if mode == "vector":
            q = table.search(vector).distance_type("cosine")
        elif mode == "text":
            q = table.search(query, query_type="fts")
        else:
            q = (
                table.search(query_type="hybrid")
                .vector(vector)
                .text(query)
                .rerank(RRFReranker())
            )
        if filters:
            q = q.where(" AND ".join(filters), prefilter=True)
        hits = q.limit(k).to_list()
        return [
            {
                key: h.get(key)
                for key in (
                    "id",
                    "kind",
                    "title",
                    "text",
                    "project",
                    "tags",
                    "source",
                    "line_start",
                    "line_end",
                    "updated_at",
                )
            }
            | {"score": _score(h)}
            for h in hits
        ]

    # -- versions ------------------------------------------------------------
    def versions(self):
        return [
            {"version": v["version"], "timestamp": str(v["timestamp"])}
            for v in self.table.list_versions()
        ]

    def restore(self, version):
        self.table.checkout(version)
        self.table.restore()
        self._reindex()
        return self.table.version
