"""BEIR SciFact: 5,183 abstracts, 300 test claims with relevance labels (CC BY-SA 4.0).

Measures nDCG@10 and recall@10 for vector, keyword (BM25) and hybrid search.

    uv run --group bench python bench/scifact.py
"""

from common import Timer, add_texts, fresh_kb, mean, ndcg, recall, table
from datasets import load_dataset


def main():
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = {
        q["_id"]: q["text"]
        for q in load_dataset("BeIR/scifact", "queries", split="queries")
    }
    qrels = {}
    for r in load_dataset("BeIR/scifact-qrels", split="test"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = r["score"]
    with fresh_kb() as kb, Timer() as t_load:
        add_texts(kb, [(d["_id"], d["title"], d["text"], 0) for d in corpus])
        rows = []
        for mode in ("vector", "text", "hybrid"):
            nd, rc = [], []
            with Timer() as t:
                for qid, gains in qrels.items():
                    ranked = [
                        h["source"] for h in kb.search(queries[qid], k=10, mode=mode)
                    ]
                    nd.append(ndcg(ranked, gains))
                    rc.append(recall(ranked, list(gains), 10))
            rows.append(
                {
                    "mode": mode,
                    "nDCG@10": f"{mean(nd):.3f}",
                    "recall@10": f"{mean(rc):.3f}",
                    "ms/query": f"{1000 * t.s / len(qrels):.0f}",
                }
            )
    print(
        f"SciFact: {len(corpus)} docs, {len(qrels)} queries, {t_load.s:.0f} s in total (load + 900 searches)"
    )
    print(table(rows, ["mode", "nDCG@10", "recall@10", "ms/query"]))


if __name__ == "__main__":
    main()
