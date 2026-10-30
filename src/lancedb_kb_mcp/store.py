"""A knowledge base in LanceDB: notes and file chunks (Markdown, text, PDF pages)
in one table with vectors, full-text search, metadata filters and versions, and
images in a second table with CLIP vectors so a text question can find them.

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
from .embed import get_embedder, get_image_embedder

TABLE = "items"
IMAGE_TABLE = "images"
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES
# CLIP scores a matching picture and its description around 0.3 and unrelated
# pairs around 0.2 to 0.27 (measured on the sample images), so below this an
# image is left out of the results instead of padding them.
IMAGE_MIN_SCORE = float(os.environ.get("LANCE_KB_IMAGE_MIN_SCORE", "0.28"))
RRF_K = 60
RESULT_KEYS = (
    "id",
    "kind",
    "title",
    "text",
    "project",
    "tags",
    "source",
    "page",
    "line_start",
    "line_end",
    "updated_at",
)


def _score(hit):
    """Higher is better in every mode: RRF relevance, BM25 score, or cosine similarity."""
    if "_relevance_score" in hit:
        return hit["_relevance_score"]
    if "_distance" in hit:
        return 1 - hit["_distance"]
    return hit.get("_score")


def _stored_hash(table, source):
    rows = (
        table.search()
        .where(f"source = {_sql(source)}")
        .select(["content_hash"])
        .limit(1)
        .to_list()
    )
    return rows[0]["content_hash"] if rows else None


def _image_result(hit):
    return {
        "id": hit["id"],
        "kind": "image",
        "title": hit["title"],
        "text": "",
        "project": hit["project"],
        "tags": [],
        "source": hit["source"],
        "page": 0,
        "line_start": 0,
        "line_end": 0,
        "updated_at": hit["updated_at"],
        "score": _score(hit) if "_distance" in hit else None,
    }


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
                pa.field("page", pa.int32()),  # PDF page, 0 otherwise
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
            if "page" not in self.table.schema.names:  # created before PDF support
                self.table.add_columns({"page": "cast(0 as int)"})
        else:
            self.table = self.db.create_table(TABLE, schema=self.schema)
        self._images = None

    @property
    def images(self):
        """The image table, or None until the first image is ingested."""
        if self._images is None and IMAGE_TABLE in self.db.list_tables().tables:
            self._images = self.db.open_table(IMAGE_TABLE)
        return self._images

    def _image_table(self, dim):
        if self.images is None:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("project", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("content_hash", pa.string()),
                    pa.field("created_at", pa.string()),
                    pa.field("updated_at", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            )
            self._images = self.db.create_table(IMAGE_TABLE, schema=schema)
        return self._images

    # -- writing -----------------------------------------------------------
    def _rows(self, items):
        vectors = self.embedder.encode([f"{i['title']}\n{i['text']}" for i in items])
        return [{**i, "vector": v} for i, v in zip(items, vectors)]

    def _reindex(self):
        if self.table.count_rows():
            # Positions let quoted text run as a phrase query instead of raising.
            self.table.create_index(
                "text", config=FTS(with_position=True), replace=True
            )

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
            "page": 0,
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
        if self.images is not None:
            self.images.delete(f"id = {_sql(item_id)}")
        self._reindex()

    def ingest_folder(self, folder, project="default", suffixes=SUFFIXES):
        """Index Markdown, text, PDF and image files under folder. Unchanged files
        are skipped; a changed file's rows are replaced. Returns counts of files
        indexed and skipped."""
        folder = Path(folder).resolve()
        indexed = skipped = 0
        for path in sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
        ):
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            # Relative to the ingested folder, so a shared KB does not carry anyone's home path.
            source = f"{folder.name}/{path.relative_to(folder).as_posix()}"
            is_image = path.suffix.lower() in IMAGE_SUFFIXES
            table = self.images if is_image else self.table
            if table is not None and _stored_hash(table, source) == digest:
                skipped += 1
                continue
            if is_image:
                self._ingest_image(path, source, digest, project)
            else:
                self._ingest_document(path, source, digest, project)
            indexed += 1
        self._reindex()
        return {"indexed": indexed, "skipped": skipped}

    def _ingest_document(self, path, source, digest, project):
        if path.suffix.lower() in PDF_SUFFIXES:
            from pypdf import PdfReader

            pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        else:
            pages = [path.read_text(encoding="utf-8", errors="replace")]
        self.table.delete(f"source = {_sql(source)}")
        now = _now()
        rows = [
            {
                "id": uuid.uuid4().hex[:12],
                "kind": "chunk",
                "title": heading
                or (f"{path.name} p.{number}" if len(pages) > 1 else path.name),
                "text": chunk,
                "project": project,
                "tags": [],
                "source": source,
                # Lines count from the top of the page for a PDF, of the file otherwise.
                "page": number if path.suffix.lower() in PDF_SUFFIXES else 0,
                "line_start": start,
                "line_end": end,
                "content_hash": digest,
                "created_at": now,
                "updated_at": now,
            }
            for number, text in enumerate(pages, 1)
            for heading, chunk, start, end in chunk_text(text)
        ]
        if rows:
            self.table.add(self._rows(rows))

    def _ingest_image(self, path, source, digest, project):
        embedder = get_image_embedder()
        table = self._image_table(embedder.dim)
        table.delete(f"source = {_sql(source)}")
        now = _now()
        table.add(
            [
                {
                    "id": uuid.uuid4().hex[:12],
                    "title": path.name,
                    "project": project,
                    "source": source,
                    "content_hash": digest,
                    "created_at": now,
                    "updated_at": now,
                    "vector": embedder.encode_images([path])[0],
                }
            ]
        )

    # -- reading -----------------------------------------------------------
    def get(self, item_id):
        rows = self.table.search().where(f"id = {_sql(item_id)}").limit(1).to_list()
        if not rows and self.images is not None:
            rows = [
                _image_result(r)
                for r in self.images.search()
                .where(f"id = {_sql(item_id)}")
                .limit(1)
                .to_list()
            ]
        return rows[0] if rows else None

    def search(
        self, query, k=8, project=None, kind=None, tag=None, version=None, mode="hybrid"
    ):
        """Hybrid (vector + full-text, merged with RRF), "vector" or "text" search.

        Images whose CLIP score clears IMAGE_MIN_SCORE are merged into the vector
        and hybrid results by rank. `version` applies to the notes and chunks table.
        """
        texts = (
            []
            if kind == "image"
            else self._search_text(query, k, project, kind, tag, version, mode)
        )
        images = (
            self._search_images(query, k, project)
            if mode != "text" and kind in (None, "image") and not tag
            else []
        )
        if not images:
            return texts
        if not texts:
            return images
        fused = {}
        for ranked in (texts, images):
            for rank, hit in enumerate(ranked, 1):
                fused[hit["id"]] = hit | {"score": 1 / (RRF_K + rank)}
        return sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]

    def _search_text(self, query, k, project, kind, tag, version, mode):
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
        if mode == "text":  # keyword search needs no embedding
            q = table.search(query, query_type="fts")
        elif mode == "vector":
            vector = self.embedder.encode([query], query=True)[0]
            q = table.search(vector).distance_type("cosine")
        else:
            vector = self.embedder.encode([query], query=True)[0]
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
            {key: h.get(key) for key in RESULT_KEYS} | {"score": _score(h)}
            for h in hits
        ]

    def _search_images(self, query, k, project):
        if self.images is None or not self.images.count_rows():
            return []
        vector = get_image_embedder().encode_text([query])[0]
        q = self.images.search(vector).distance_type("cosine")
        if project:
            q = q.where(f"project = {_sql(project)}", prefilter=True)
        hits = [_image_result(h) for h in q.limit(k).to_list()]
        return [h for h in hits if h["score"] >= IMAGE_MIN_SCORE]

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
