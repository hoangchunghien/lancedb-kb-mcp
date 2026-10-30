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


def _pdf(path, pages):
    """Write a minimal PDF with one line of Helvetica text per page."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream"
        )
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def test_pdf_chunks_keep_their_page(kb, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _pdf(
        docs / "postmortem.pdf",
        ["Summary of the outage", "Root cause: connection pool exhausted"],
    )
    assert kb.ingest_folder(str(docs)) == {"indexed": 1, "skipped": 0}
    hit = kb.search("connection pool", mode="text")[0]
    assert (hit["source"], hit["page"], hit["title"]) == (
        "docs/postmortem.pdf",
        2,
        "postmortem.pdf p.2",
    )
    assert kb.ingest_folder(str(docs)) == {"indexed": 0, "skipped": 1}


def test_images_are_found_by_text_and_merged(kb, tmp_path):
    from PIL import Image

    docs = tmp_path / "docs"
    docs.mkdir()
    Image.new("RGB", (8, 8), "white").save(docs / "queue-depth-chart.png")
    (docs / "runbook.md").write_text(
        "# Queue backlog\nCheck the consumer error rate first.\n"
    )
    assert kb.ingest_folder(str(docs)) == {"indexed": 2, "skipped": 0}
    hits = kb.search("queue depth chart")
    assert {h["kind"] for h in hits} == {"image", "chunk"}
    image = next(h for h in hits if h["kind"] == "image")
    assert image["source"] == "docs/queue-depth-chart.png"
    assert kb.get(image["id"])["kind"] == "image"
    assert kb.search("queue depth chart", kind="image")[0]["id"] == image["id"]
    assert all(
        h["kind"] != "image" for h in kb.search("queue depth chart", mode="text")
    )
    assert kb.ingest_folder(str(docs)) == {"indexed": 0, "skipped": 2}
    kb.delete(image["id"])
    assert kb.get(image["id"]) is None


def test_quoted_query_is_a_phrase_search(kb):
    kb.add_note("Backlog", "The queue backlog alert fires after ten minutes.")
    kb.add_note("Other", "The backlog of the queue is long.")
    hits = kb.search('"queue backlog"', mode="text")
    assert [h["title"] for h in hits] == ["Backlog"]
    assert kb.search('what does "queue backlog" mean')  # hybrid search with quotes
