"""ViDoRe V3, computer-science set: does the right PDF page come back?

The corpus is OpenStax "Introduction to Computer Science", the 945-page edition the
benchmark was built from (Internet Archive copy, December 2024), ingested through the
same PDF path as any other file. 132 of the 215 English questions are labelled only on
this book and are used here; the other textbook's labelled edition is no longer
published, so its 79 questions are left out.

Two runs over the same pages and questions:
  pdf       our ingest: pypdf text per page, chunked, page kept
  markdown  ViDoRe's own per-page Markdown, one chunk per page (how much extraction costs)

    uv run --group bench python bench/vidore_pdf.py
"""

import os
import urllib.request

from common import CACHE, Timer, add_texts, fresh_kb, hit, mean, ndcg, table
from datasets import load_dataset

DATASET = "vidore/vidore_v3_computer_science"
BOOK = "Introduction_to_Computer_Science"
PDF_URL = (
    "https://web.archive.org/web/20241214021836id_/https://assets.openstax.org/"
    "oscms-prodcms/media/documents/Introduction_To_Computer_Science_-_WEB.pdf"
)


def labels():
    corpus = load_dataset(
        DATASET,
        "corpus",
        split="test",
        columns=["corpus_id", "doc_id", "page_number_in_doc", "markdown"],
    )
    page_of = {
        c["corpus_id"]: (c["doc_id"], c["page_number_in_doc"] + 1) for c in corpus
    }  # 1-based, like ours
    queries = {
        q["query_id"]: q["query"]
        for q in load_dataset(DATASET, "queries", split="test")
        if q["language"] == "english"
    }
    gains = {}
    for r in load_dataset(DATASET, "qrels", split="test"):
        if r["query_id"] in queries:
            gains.setdefault(r["query_id"], {})[page_of[r["corpus_id"]]] = r["score"]
    only_book = {
        q: {p: g for (d, p), g in gs.items()}
        for q, gs in gains.items()
        if {d for d, _ in gs} == {BOOK}
    }
    markdown = [
        (
            f"{BOOK}.pdf",
            f"p.{c['page_number_in_doc'] + 1}",
            c["markdown"],
            c["page_number_in_doc"] + 1,
        )
        for c in corpus
        if c["doc_id"] == BOOK and c["markdown"].strip()
    ]
    return queries, only_book, markdown


def pages(kb, query):
    """Rank pages by their best chunk: the first time a page shows up in the chunk ranking."""
    seen = []
    for h in kb.search(query, k=60):
        if h["page"] not in seen:
            seen.append(h["page"])
    return seen


def run(kb, queries, gains):
    nd, h1, h5, h10 = [], [], [], []
    with Timer() as t:
        for q, g in gains.items():
            ranked = pages(kb, queries[q])
            nd.append(ndcg(ranked, g))
            h1.append(hit(ranked, g, 1))
            h5.append(hit(ranked, g, 5))
            h10.append(hit(ranked, g, 10))
    return {
        "nDCG@10": f"{mean(nd):.3f}",
        "hit@1": f"{mean(h1):.2f}",
        "hit@5": f"{mean(h5):.2f}",
        "hit@10": f"{mean(h10):.2f}",
        "ms/query": f"{1000 * t.s / len(gains):.0f}",
    }


def main():
    queries, gains, markdown = labels()
    folder = os.path.join(CACHE, "vidore_cs")
    os.makedirs(folder, exist_ok=True)
    pdf = os.path.join(folder, f"{BOOK}.pdf")
    if not os.path.exists(pdf):
        urllib.request.urlretrieve(PDF_URL, pdf)
    rows = []
    with fresh_kb() as kb:
        with Timer() as t:
            kb.ingest_folder(folder, "bench")
        chunks = kb.table.count_rows()
        rows.append(
            {"corpus": f"pdf (pypdf, {chunks} chunks)", **run(kb, queries, gains)}
        )
        ingest_s = t.s
    with fresh_kb() as kb:
        add_texts(kb, markdown)
        rows.append(
            {"corpus": f"markdown ({len(markdown)} pages)", **run(kb, queries, gains)}
        )
    print(
        f"ViDoRe V3 CS: {len(gains)} English questions on a 945-page PDF; PDF ingest {ingest_s:.0f} s"
    )
    print("hit@k = a relevant page is in the top k pages; hybrid search")
    print(table(rows, ["corpus", "nDCG@10", "hit@1", "hit@5", "hit@10", "ms/query"]))


if __name__ == "__main__":
    main()
