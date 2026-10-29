"""MCP server: exposes the knowledge base to Claude (or any MCP client) over stdio."""

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from .store import KnowledgeBase

mcp = MCPServer(
    "lancedb-kb",
    instructions=(
        "A local knowledge base stored in LanceDB. Call search before answering questions about "
        "the user's notes or indexed documents, and cite the source and line range of what you use. "
        "Save decisions and findings worth keeping with add_note."
    ),
)


@lru_cache(maxsize=1)
def kb() -> KnowledgeBase:
    # Opened on first use, so the server starts instantly and loads the model only when needed.
    return KnowledgeBase()


@mcp.tool()
def search(
    query: str,
    k: int = 8,
    project: str | None = None,
    kind: str | None = None,
    tag: str | None = None,
    version: int | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    """Search notes and document chunks. mode is "hybrid" (vector + keyword, merged with RRF),
    "vector" or "text". Filter by project, kind ("note" or "chunk") or tag. Pass version to
    search the knowledge base as it was at that version (see list_versions)."""
    return kb().search(
        query, k=k, project=project, kind=kind, tag=tag, version=version, mode=mode
    )


@mcp.tool()
def get(item_id: str) -> dict | None:
    """Fetch one note or chunk by id, with its full text and metadata."""
    item = kb().get(item_id)
    return None if item is None else {k: v for k, v in item.items() if k != "vector"}


@mcp.tool()
def add_note(
    title: str, text: str, project: str = "default", tags: list[str] | None = None
) -> str:
    """Save a note (a decision, a finding, a how-to). Returns its id."""
    return kb().add_note(title, text, project, tags or [])


@mcp.tool()
def update_note(
    item_id: str,
    title: str | None = None,
    text: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Change a note's title, text, project or tags. It is re-embedded automatically."""
    return kb().update_note(item_id, title, text, project, tags)


@mcp.tool()
def delete(item_id: str) -> str:
    """Delete a note or chunk. It stays recoverable through restore_version."""
    kb().delete(item_id)
    return "deleted"


@mcp.tool()
def ingest_folder(path: str, project: str = "default") -> dict:
    """Index the Markdown, text and reStructuredText files under path, chunked by heading with
    file and line provenance. Unchanged files are skipped; changed files are re-chunked."""
    return kb().ingest_folder(path, project)


@mcp.tool()
def list_versions() -> list[dict]:
    """List the knowledge base's versions (every write creates one)."""
    return kb().versions()


@mcp.tool()
def restore_version(version: int) -> int:
    """Make an older version current again (a new version is written; history is kept)."""
    return kb().restore(version)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
