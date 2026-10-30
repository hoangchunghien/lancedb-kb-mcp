"""Shared pieces for the benchmarks: metrics, a fresh knowledge base, bulk loading."""

import hashlib
import math
import os
import tempfile
import time
import uuid
from contextlib import contextmanager

from lancedb_kb_mcp.store import KnowledgeBase

CACHE = os.environ.get("BENCH_CACHE", os.path.join(os.path.dirname(__file__), ".cache"))


def ndcg(ranked, gains, k=10):
    """nDCG@k with graded gains: {doc: gain}."""
    dcg = sum(gains.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sum(
        g / math.log2(i + 2)
        for i, g in enumerate(sorted(gains.values(), reverse=True)[:k])
    )
    return dcg / ideal if ideal else 0.0


def recall(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def hit(ranked, relevant, k):
    return float(bool(set(ranked[:k]) & set(relevant)))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


@contextmanager
def fresh_kb(embedder=None):
    with tempfile.TemporaryDirectory() as d:
        yield KnowledgeBase(os.path.join(d, "kb"), embedder=embedder)


def add_texts(kb, docs, batch=256):
    """Load (source, title, text, page) tuples as chunks, without file I/O."""
    for i in range(0, len(docs), batch):
        rows = [
            {
                "id": uuid.uuid4().hex[:12],
                "kind": "chunk",
                "title": title,
                "text": text,
                "project": "bench",
                "tags": [],
                "source": source,
                "page": page,
                "line_start": 0,
                "line_end": 0,
                "content_hash": hashlib.sha1(text.encode()).hexdigest(),
                "created_at": "",
                "updated_at": "",
            }
            for source, title, text, page in docs[i : i + batch]
        ]
        kb.table.add(kb._rows(rows))
    kb._reindex()


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
    return "\n".join(out)
