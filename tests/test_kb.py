import pytest

from lancedb_kb_mcp.embed import HashEmbedder
from lancedb_kb_mcp.store import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(str(tmp_path / "kb"), embedder=HashEmbedder())


def test_note_round_trip_and_filters(kb):
    a = kb.add_note(
        "Retry policy", "Retry 5xx only, with a capped backoff.", "payments", ["retry"]
    )
    kb.add_note(
        "Error E1042", "E1042 means the upload token expired.", "uploads", ["errors"]
    )
    assert kb.get(a)["title"] == "Retry policy"
    assert kb.search("E1042", mode="text")[0]["title"] == "Error E1042"
    assert {h["project"] for h in kb.search("retry", project="payments")} == {
        "payments"
    }
    assert all("retry" in h["tags"] for h in kb.search("backoff", tag="retry"))


def test_update_reembeds(kb):
    a = kb.add_note("Old", "first text")
    kb.update_note(a, text="completely different words about caching")
    assert kb.search("caching", mode="text")[0]["id"] == a


def test_ingest_skips_unchanged_and_replaces_changed(kb, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    f = docs / "a.md"
    f.write_text("# Intro\nhello\n\n# Setup\nrun the installer\n")
    assert kb.ingest_folder(str(docs)) == {"indexed": 1, "skipped": 0}
    assert kb.ingest_folder(str(docs)) == {"indexed": 0, "skipped": 1}
    hit = kb.search("installer", mode="text")[0]
    assert (hit["source"].endswith("a.md"), hit["line_start"]) == (True, 4)
    f.write_text("# Setup\nuse the container image\n")
    kb.ingest_folder(str(docs))
    assert kb.search("installer", mode="text", kind="chunk") == []


def test_versions_time_travel_and_restore(kb):
    a = kb.add_note("Keep", "durable fact")
    before = kb.versions()[-1]["version"]
    kb.delete(a)
    assert kb.get(a) is None
    assert kb.search("durable", mode="text", version=before)[0]["id"] == a
    kb.restore(before)
    assert kb.get(a)["title"] == "Keep"


def test_rst_headings_split_chunks():
    from lancedb_kb_mcp.chunking import chunk_text

    text = "Intro\n=====\n\nhello\n\nCheck argument types [arg-type]\n-------------------------------\n\nbody\n"
    chunks = list(chunk_text(text))
    assert [c[0] for c in chunks] == ["Intro", "Check argument types [arg-type]"]
    assert chunks[1][2] == 6
