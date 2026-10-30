"""Does a small language model before the search help?

Two ways to put a model in front of the same hybrid search (bge-small):
  rewrite  the model turns the question into a short keyword query
  hyde     the model writes a brief answer, and the search runs on question + answer
           (Hypothetical Document Embeddings: the made-up answer reads like the documents)

Measured on SciFact (nDCG@10) and the ViDoRe PDF (right page first), plus the time the
model adds per question.

    BENCH_LLM=Qwen/Qwen2.5-0.5B-Instruct BENCH_LLM_DEVICE=cuda uv run --group bench python bench/presearch.py
"""

import json
import os
import statistics
import time

import models
import torch
from common import add_texts, fresh_kb, hit, mean, ndcg
from transformers import AutoModelForCausalLM, AutoTokenizer

from lancedb_kb_mcp import embed

LLM = os.environ.get("BENCH_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get(
    "BENCH_LLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
PROMPTS = {
    "rewrite": (
        "Rewrite the question as a short search query of key terms. Reply with the query only.",
        32,
    ),
    "hyde": (
        "Write a short factual passage, two or three sentences, that answers the question. Reply with the passage only.",
        96,
    ),
}
OUT = os.path.join(
    os.path.dirname(__file__), "results", f"presearch-{LLM.split('/')[-1]}.json"
)


class Generator:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(LLM, padding_side="left")
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(LLM, dtype=dtype).to(DEVICE).eval()
        )

    def __call__(self, kind, questions, batch=16):
        system, max_new = PROMPTS[kind]
        out = []
        for i in range(0, len(questions), batch):
            chats = [
                self.tok.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": q},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in questions[i : i + batch]
            ]
            enc = self.tok(chats, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
            out += [
                t.strip()
                for t in self.tok.batch_decode(
                    gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
            ]
        return out


def query_text(kind, question, generated):
    return generated if kind == "rewrite" else f"{question}\n{generated}"


def evaluate(kb_sf, kb_vd, sf, vd, texts_sf, texts_vd):
    (_, _, sqrels), (_, _, vgains) = sf, vd
    nd = mean(
        ndcg([h["source"] for h in kb_sf.search(texts_sf[q], k=10)], g)
        for q, g in sqrels.items()
    )
    h1, h5 = [], []
    for q, g in vgains.items():
        seen = []
        for h in kb_vd.search(texts_vd[q], k=60):
            if h["page"] not in seen:
                seen.append(h["page"])
        h1.append(hit(seen, g, 1))
        h5.append(hit(seen, g, 5))
    return {
        "scifact_ndcg10": round(nd, 4),
        "vidore_hit1": round(mean(h1), 4),
        "vidore_hit5": round(mean(h5), 4),
    }


def main():
    sf, vd = models.scifact_data(), models.vidore_data()
    sq = {q: sf[1][q] for q in sf[2]}
    vq = {q: vd[1][q] for q in vd[2]}
    gen = Generator()
    os.environ["LANCE_KB_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    e = embed.SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    out = {"llm": LLM, "device": DEVICE, "search": "bge-small hybrid", "examples": {}}
    with fresh_kb(e) as kb_sf, fresh_kb(e) as kb_vd:
        add_texts(kb_sf, sf[0])
        add_texts(kb_vd, vd[0])
        out["none"] = evaluate(kb_sf, kb_vd, sf, vd, sq, vq)
        print("none", out["none"], flush=True)
        for kind in PROMPTS:
            t0 = time.perf_counter()
            g_sf = dict(zip(sq, gen(kind, list(sq.values()))))
            g_vd = dict(zip(vq, gen(kind, list(vq.values()))))
            batch_s = time.perf_counter() - t0
            texts_sf = {q: query_text(kind, sq[q], g_sf[q]) for q in sq}
            texts_vd = {q: query_text(kind, vq[q], g_vd[q]) for q in vq}
            row = evaluate(kb_sf, kb_vd, sf, vd, texts_sf, texts_vd)
            # Time one question at a time, as a search would run it.
            one = []
            for q in list(sq.values())[:20]:
                t = time.perf_counter()
                gen(kind, [q], batch=1)
                one.append(1000 * (time.perf_counter() - t))
            row["ms_per_question"] = round(statistics.median(one))
            row["batch_s"] = round(batch_s)
            out[kind] = row
            first = next(iter(sq))
            out["examples"][kind] = {"question": sq[first], "generated": g_sf[first]}
            print(kind, row, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
