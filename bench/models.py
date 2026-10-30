"""Compare local embedding models on the same benchmarks, and save the numbers for charts.

Text models run SciFact and the ViDoRe PDF set (vector and hybrid search); CLIP models run
Flickr30k. Each model uses the query and document prompts its model card recommends.
Quality runs on the GPU when there is one (scores do not depend on the device); query
latency is measured on the CPU in float32, one question at a time, as on a laptop.

    uv run --group bench python bench/models.py          # writes bench/results/models.json
    uv run --group bench python bench/charts.py          # draws the SVG charts
"""

import gc
import json
import os
import statistics
import time
from functools import partial

import flickr_images
import scifact  # noqa: F401  (keeps the dataset loaders in one place)
import torch
import vidore_pdf
from common import CACHE, Timer, add_texts, fresh_kb, hit, mean, ndcg, recall
from datasets import load_dataset

from lancedb_kb_mcp import embed, store
from lancedb_kb_mcp.chunking import chunk_text

TEXT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "Snowflake/snowflake-arctic-embed-s",
    "intfloat/e5-small-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/multilingual-e5-small",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Qwen/Qwen3-Embedding-0.6B",
]
# Qwen3-Embedding-8B as served for a personal knowledge base: llama.cpp, Q6_K GGUF, last-token
# pooling, output cut to 1536 dimensions and re-normalised (Qwen3 embeddings are MRL-trained).
# Read-only requests to a running server; set LIVE_EMBED_URL to use it.
LIVE_8B = "Qwen3-Embedding-8B (Q6_K, llama.cpp, 1536 dims)"
LIVE_URL = os.environ.get("LIVE_EMBED_URL", "")
if LIVE_URL:
    TEXT_MODELS.append(LIVE_8B)
IMAGE_MODELS = ["clip-ViT-B-32", "clip-ViT-B-16", "clip-ViT-L-14"]
OUT = os.path.join(os.path.dirname(__file__), "results", "models.json")
GPU = "cuda" if torch.cuda.is_available() else "cpu"


class LlamaServerEmbedder:
    """Embeddings from a llama.cpp server's /v1/embeddings, the way a Qwen3 client calls it."""

    name = LIVE_8B
    dim = 1536
    prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

    def __init__(self, url=None, batch=16):
        import httpx

        self.client = httpx.Client(base_url=url or LIVE_URL, timeout=300)
        self.batch = batch

    def encode(self, texts, query=False):
        texts = [(self.prefix + t if query else t)[:12000] for t in texts]
        out = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.post(
                "/v1/embeddings",
                json={"input": texts[i : i + self.batch], "model": "embedding"},
            )
            resp.raise_for_status()
            for d in sorted(resp.json()["data"], key=lambda d: d["index"]):
                v = d["embedding"][: self.dim]
                norm = sum(x * x for x in v) ** 0.5 or 1.0
                out.append([x / norm for x in v])
        return out


def embedder(name, device, image=False):
    os.environ["LANCE_KB_DEVICE"] = device
    if image:
        return embed.ClipEmbedder(name)
    e = embed.SentenceTransformerEmbedder(name)
    # Most of these models stop at 512 tokens; hold the long-context ones (Qwen3) to the same
    # limit so the comparison is like for like and fits beside other GPU work.
    e.model.max_seq_length = min(e.model.max_seq_length or 512, 512)
    return e


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def params(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6)


def cpu_latency(fn, items):
    fn(items[:3])  # warm up
    times = []
    for item in items:
        t = time.perf_counter()
        fn([item])
        times.append(1000 * (time.perf_counter() - t))
    return round(statistics.median(times), 1)


def scifact_data():
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = {
        q["_id"]: q["text"]
        for q in load_dataset("BeIR/scifact", "queries", split="queries")
    }
    qrels = {}
    for r in load_dataset("BeIR/scifact-qrels", split="test"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = r["score"]
    return [(d["_id"], d["title"], d["text"], 0) for d in corpus], queries, qrels


def vidore_data():
    from pypdf import PdfReader

    queries, gains, _ = vidore_pdf.labels()
    pdf = os.path.join(CACHE, "vidore_cs", f"{vidore_pdf.BOOK}.pdf")
    name = os.path.basename(pdf)
    # The same steps as the store's PDF ingest: text per page, chunked, page kept.
    docs = [
        (name, heading or f"{name} p.{n}", chunk, n)
        for n, page in enumerate(PdfReader(pdf).pages, 1)
        for heading, chunk, _, _ in chunk_text(page.extract_text() or "")
    ]
    return docs, queries, gains


def sample_queries(sf):
    return list(sf[1].values())[:100]


def text_scores(kb, data, modes=("vector", "hybrid")):
    (_, queries, qrels), out = data, {}
    for mode in modes:
        nd, rc = [], []
        for q, gains in qrels.items():
            ranked = [h["source"] for h in kb.search(queries[q], k=10, mode=mode)]
            nd.append(ndcg(ranked, gains))
            rc.append(recall(ranked, list(gains), 10))
        out[mode] = {"ndcg10": round(mean(nd), 4), "recall10": round(mean(rc), 4)}
    return out


def page_scores(kb, data, modes=("vector", "hybrid")):
    (_, queries, gains), out = data, {}
    for mode in modes:
        nd, h1, h5 = [], [], []
        for q, g in gains.items():
            seen = []
            for h in kb.search(queries[q], k=60, mode=mode):
                if h["page"] not in seen:
                    seen.append(h["page"])
            nd.append(ndcg(seen, g))
            h1.append(hit(seen, g, 1))
            h5.append(hit(seen, g, 5))
        out[mode] = {
            "ndcg10": round(mean(nd), 4),
            "hit1": round(mean(h1), 4),
            "hit5": round(mean(h5), 4),
        }
    return out


def save(results):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)


def text_model(name, sf, vd, sample):
    e = LlamaServerEmbedder() if name == LIVE_8B else embedder(name, GPU)
    row = {
        "model": name,
        "params_m": 8000 if name == LIVE_8B else params(e.model),
        "dim": e.dim,
    }
    with fresh_kb(e) as kb:
        with Timer() as t:
            add_texts(kb, sf[0])
        row["scifact_ingest_s"] = round(t.s, 1)
        row["scifact"] = text_scores(kb, sf)
    with fresh_kb(e) as kb:
        add_texts(kb, vd[0])
        row["vidore"] = page_scores(kb, vd)
    if (
        name == LIVE_8B
    ):  # too big for a laptop CPU; time it where it runs, on the GPU server
        row["gpu_server_query_ms"] = cpu_latency(partial(e.encode, query=True), sample)
        return row
    del e
    free()
    cpu = embedder(name, "cpu")
    row["cpu_query_ms"] = cpu_latency(partial(cpu.encode, query=True), sample)
    del cpu
    free()
    return row


def image_model(name, photos, captions):
    os.environ["LANCE_KB_IMAGE_MODEL"] = name
    os.environ["LANCE_KB_DEVICE"] = GPU
    store.IMAGE_MIN_SCORE = -1.0  # rank every photo
    embed.get_image_embedder.cache_clear()
    e = embed.get_image_embedder()
    row = {"model": name, "params_m": params(e.model), "dim": e.dim}
    with fresh_kb(embed.HashEmbedder()) as kb:
        with Timer() as t:
            kb.ingest_folder(photos, "bench")
        row["ingest_s"] = round(t.s, 1)
        r1, r5, r10 = [], [], []
        for caption, filename in captions:
            names = [h["title"] for h in kb.search(caption, k=10, kind="image")]
            r1.append(filename in names[:1])
            r5.append(filename in names[:5])
            r10.append(filename in names[:10])
    row.update(
        recall1=round(mean(r1), 4),
        recall5=round(mean(r5), 4),
        recall10=round(mean(r10), 4),
    )
    del e
    embed.get_image_embedder.cache_clear()
    free()
    cpu = embedder(name, "cpu", image=True)
    row["cpu_query_ms"] = cpu_latency(cpu.encode_text, [c for c, _ in captions[:100]])
    del cpu
    free()
    return row


def main():
    # Results are saved after every model, and a re-run skips the models already done.
    results = {"text": [], "image": [], "keyword": {}, "skipped": []}
    if os.path.exists(OUT):
        with open(OUT) as f:
            results.update(json.load(f))
    results["device"] = torch.cuda.get_device_name(0) if GPU == "cuda" else "cpu"
    done = {r["model"] for r in results["text"] + results["image"] + results["skipped"]}
    only = os.environ.get(
        "BENCH_ONLY"
    )  # run one model, in its own process, e.g. to free GPU memory
    if only:
        done |= {m for m in TEXT_MODELS + IMAGE_MODELS if m != only}
    sf, vd = scifact_data(), vidore_data()
    with fresh_kb(embed.HashEmbedder()) as kb:  # keyword search does not use the model
        add_texts(kb, sf[0])
        results["keyword"]["scifact"] = text_scores(kb, sf, ("text",))["text"]
        # Search time without any model: keyword only, and the hybrid search's own cost
        # (the hashing embedder takes microseconds), to add to each model's embed time.
        results["keyword"]["search_ms"] = cpu_latency(
            lambda qs: kb.search(qs[0], k=10, mode="text"), sample_queries(sf)
        )
        results["keyword"]["hybrid_search_ms"] = cpu_latency(
            lambda qs: kb.search(qs[0], k=10, mode="hybrid"), sample_queries(sf)
        )
    with fresh_kb(embed.HashEmbedder()) as kb:
        add_texts(kb, vd[0])
        results["keyword"]["vidore"] = page_scores(kb, vd, ("text",))["text"]
    save(results)
    sample = sample_queries(sf)
    for name in TEXT_MODELS:
        if name in done:
            continue
        print("text model", name, flush=True)
        try:
            row = text_model(name, sf, vd, sample)
        # A model whose remote code no longer runs on current transformers is recorded, not fatal.
        except Exception as exc:  # noqa: BLE001
            print("  skipped:", type(exc).__name__, exc, flush=True)
            results["skipped"].append(
                {"model": name, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            free()
            save(results)
            continue
        print(" ", json.dumps(row), flush=True)
        results["text"].append(row)
        save(results)
    photos, captions = flickr_images.photos()
    for name in IMAGE_MODELS:
        if name in done:
            continue
        print("image model", name, flush=True)
        row = image_model(name, photos, captions)
        print(" ", json.dumps(row), flush=True)
        results["image"].append(row)
        save(results)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
