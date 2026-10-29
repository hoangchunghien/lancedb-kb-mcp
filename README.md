# lancedb-kb-mcp

A knowledge base for Claude (or any MCP client) that lives in one folder on disk.

- **Hybrid search**: vector similarity and BM25 keyword search in one query, merged with
  reciprocal rank fusion. Meaning-based questions and exact strings like `E1042` both work.
- **Notes**: Claude can save, update and delete notes while it works.
- **Folder ingest**: Markdown, text and reStructuredText files are chunked by heading, with
  the file and line range kept on every chunk so answers can cite their source.
- **Versions**: every write creates a new version. Search the knowledge base as it was
  yesterday, or roll it back.

No server to run, no Docker, no API key. LanceDB is an embedded database (like SQLite, but for
vectors), and the embedding model is small enough to run on a laptop CPU.

## How it fits together

```
Claude Code ──stdio──▶ lancedb-kb-mcp ──▶ LanceDB table in ./kb-data
                           │                 ├─ vectors (384-dim)
                           │                 ├─ full-text index
                           └─ bge-small      └─ version history
                              (local model)
```

One table holds both notes and document chunks. Each row has the text, a title, a project, tags,
the source file and line range, and its vector. Writes go through LanceDB's `merge_insert`, and
the full-text index is rebuilt after each write.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```bash
git clone <this repo> && cd lancedb-kb-mcp
uv sync

# Index the sample docs and try a search from the command line
uv run lance-kb ingest examples/docs --project demo
uv run lance-kb search "why is the queue backing up"
uv run lance-kb search "E1042"
```

The first run downloads the embedding model (about 130 MB) into the Hugging Face cache. After
that, ingesting the three sample files takes a few seconds.

Notice that the first query shares no keywords with the runbook it finds, and the second is an
error code that a vector search alone tends to miss. Hybrid search handles both.

## Use it from Claude Code

This repo ships a project-scoped `.mcp.json`, so the quickest trial is to open Claude Code in this
folder:

```bash
claude
```

Approve the `lance-kb` server when asked, then try:

- "Index the docs in examples/docs under project demo."
- "What should I check first when the queue backlog alert fires?"
- "Save a note: device cache entries expire after 15 minutes, tag it cache."
- "List the knowledge base versions, then search for E1042 as it was at version 2."

To use the same knowledge base from any project, register it once at user scope:

```bash
claude mcp add lance-kb --scope user \
  --env LANCE_KB_DIR=$HOME/lance-kb-data \
  -- uv run --directory /path/to/lancedb-kb-mcp lancedb-kb-mcp
```

Any other MCP client works the same way: the command is `uv run lancedb-kb-mcp`, over stdio.

## Tools

| Tool | What it does |
|---|---|
| `search(query, k, project, kind, tag, version, mode)` | Hybrid search by default; `mode` can be `vector` or `text`. Filters apply before the search, so a narrow filter still returns `k` results. `version` searches an older version. |
| `get(item_id)` | One note or chunk with its full text and metadata. |
| `add_note(title, text, project, tags)` | Save a note and return its id. |
| `update_note(item_id, ...)` | Change a note; it is re-embedded. |
| `delete(item_id)` | Delete a note or chunk. It stays in the version history. |
| `ingest_folder(path, project)` | Index a folder. Unchanged files are skipped by content hash; changed files have their chunks replaced. |
| `list_versions()` | Every version with its timestamp. |
| `restore_version(version)` | Make an older version current. This writes a new version, so nothing is lost. |

Every search result has a `score` where higher is better: the fused rank score in hybrid mode,
BM25 in text mode, cosine similarity in vector mode.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LANCE_KB_DIR` | `kb-data` | Where the LanceDB table lives. |
| `LANCE_KB_MODEL` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model. `hash` uses a tiny bag-of-words embedder with no download, meant for tests. |
| `LANCE_KB_DEVICE` | auto | `cpu`, `cuda` or `mps`. |

The model is part of your schema. Vectors from different models are not comparable, so if you
change `LANCE_KB_MODEL`, start a new `LANCE_KB_DIR` and re-ingest.

Why bge-small: 33M parameters and 384 dimensions, so it runs comfortably on a laptop CPU and
keeps the table small, while scoring well on retrieval benchmarks for its size. For non-English
content, swap in a multilingual model such as `intfloat/multilingual-e5-small`.

## Sharing it with a team

The knowledge base is just a folder, so the simplest way to share it is to share the folder:
commit a snapshot, sync it, or copy it. Each person points `LANCE_KB_DIR` at their copy.

LanceDB can also open a table on object storage (`s3://`, `gs://`, `az://`), which lets a team
read the same table. This demo has only been tested on a local folder. If you try a shared
bucket, keep to one writer at a time (for example, a scheduled job that runs `lance-kb ingest`
on your docs) and let everyone else read.

## Development

```bash
LANCE_KB_MODEL=hash uv run pytest
uv run ruff check src tests
```

The tests use the hash embedder, so they run in about a second with no model download.

## Limits

- One process writing at a time. Two Claude sessions writing notes to the same local folder can
  race; reads are fine.
- The full-text index is rebuilt after every write, which is fine for thousands of rows and slow
  for millions. A larger deployment would batch writes and reindex on a schedule.
- No vector index is built. Exact search over tens of thousands of chunks is fast; past a few
  hundred thousand, add one with `table.create_index(...)`.
- Only text files are ingested. PDFs, images and code need their own chunkers.
