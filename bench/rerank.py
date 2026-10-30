"""Does a reranker help, and what does it cost?

First stage: bge-small hybrid search, top 50 candidates. Second stage: a cross-encoder reads
the question with each candidate and reorders them. Measured on SciFact (nDCG@10) and the
ViDoRe PDF (right page first), with the cost on a CPU: 50 question-passage pairs per question.

    uv run --group bench python bench/rerank.py      # writes bench/results/rerank.json
"""

import json
import os
import statistics
import time

import models
import torch
from common import add_texts, fresh_kb, hit, mean, ndcg
from sentence_transformers import CrossEncoder

from lancedb_kb_mcp import embed

RERANKERS = ["cross-encoder/ms-marco-MiniLM-L6-v2", "BAAI/bge-reranker-base"]
CANDIDATES = 50
OUT = os.path.join(os.path.dirname(__file__), "results", "rerank.json")
GPU = "cuda" if torch.cuda.is_available() else "cpu"


def candidates(kb, queries, ids):
    return {q: kb.search(queries[q], k=CANDIDATES, mode="hybrid") for q in ids}


def rerank(ce, query, hits):
    if not hits:
        return hits
    scores = ce.predict(
        [(query, f"{h['title']}\n{h['text']}") for h in hits], show_progress_bar=False
    )
    return [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]


def pages(hits):
    seen = []
    for h in hits:
        if h["page"] not in seen:
            seen.append(h["page"])
    return seen


def score(sf_c, vd_c, sf, vd, order):
    (_, sq, sqrels), (_, vq, vgains) = sf, vd
    nd = mean(
        ndcg([h["source"] for h in order(sq[q], sf_c[q])][:10], g)
        for q, g in sqrels.items()
    )
    h1 = mean(hit(pages(order(vq[q], vd_c[q])), g, 1) for q, g in vgains.items())
    h5 = mean(hit(pages(order(vq[q], vd_c[q])), g, 5) for q, g in vgains.items())
    return {
        "scifact_ndcg10": round(nd, 4),
        "vidore_hit1": round(h1, 4),
        "vidore_hit5": round(h5, 4),
    }


def cpu_ms(name, sf, sf_c):
    ce = CrossEncoder(name, device="cpu")
    qs = list(sf[2])[:30]
    ce.predict([("warm up", "warm up")])
    times = []
    for q in qs:
        t = time.perf_counter()
        rerank(ce, sf[1][q], sf_c[q])
        times.append(1000 * (time.perf_counter() - t))
    return round(statistics.median(times))


def main():
    sf, vd = models.scifact_data(), models.vidore_data()
    os.environ["LANCE_KB_DEVICE"] = GPU
    e = embed.SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    with fresh_kb(e) as kb:
        add_texts(kb, sf[0])
        sf_c = candidates(kb, sf[1], sf[2])
    with fresh_kb(e) as kb:
        add_texts(kb, vd[0])
        vd_c = candidates(kb, vd[1], vd[2])
    out = {
        "first_stage": "bge-small-en-v1.5 hybrid, top 50",
        "candidates": CANDIDATES,
        "none": score(sf_c, vd_c, sf, vd, lambda q, hits: hits),
        "rerankers": [],
    }
    print("no rerank", out["none"], flush=True)
    for name in RERANKERS:
        ce = CrossEncoder(name, device=GPU)
        row = {
            "model": name,
            **score(sf_c, vd_c, sf, vd, lambda q, hits, ce=ce: rerank(ce, q, hits)),
        }
        del ce
        models.free()
        row["cpu_ms_per_question"] = cpu_ms(name, sf, sf_c)
        print(row, flush=True)
        out["rerankers"].append(row)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
