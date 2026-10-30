"""Flickr30k 1k test: can a text question find the right photo?

1,000 photos with 5 captions each (the standard Karpathy test split, as packaged in
nlphuji/flickr_1k_test_image_text_retrieval; Flickr terms, research use). The photos go
through the normal image ingest (CLIP), and every caption is a query.

Also checks the 0.28 cut-off that keeps images out of unrelated results:
  - how many correct photos score below it (would be hidden), and
  - how many SciFact claims (text that is not about any photo) would surface a photo.

    uv run --group bench python bench/flickr_images.py
"""

import os

# Rank every photo; the cut-off is measured below. Set before the store module reads it.
os.environ["LANCE_KB_IMAGE_MIN_SCORE"] = "-1"

from common import CACHE, Timer, fresh_kb, mean, table
from datasets import load_dataset

from lancedb_kb_mcp.store import IMAGE_MIN_SCORE

CUTOFF = 0.28  # the shipped default


def photos():
    """Save the 1,000 test photos to a folder once; return it and the (caption, file) pairs."""
    # The repo uses a loading script, which current `datasets` refuses; read Hugging Face's
    # Parquet export instead.
    data = load_dataset(
        "parquet",
        split="train",
        data_files=(
            "https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval/"
            "resolve/refs%2Fconvert%2Fparquet/TEST/test/0000.parquet"
        ),
    )
    folder = os.path.join(CACHE, "flickr1k")
    os.makedirs(folder, exist_ok=True)
    for row in data:
        path = os.path.join(folder, row["filename"])
        if not os.path.exists(path):
            row["image"].convert("RGB").save(path, quality=92)
    return folder, [(c, row["filename"]) for row in data for c in row["caption"]]


def main():
    folder, captions = photos()
    assert IMAGE_MIN_SCORE < 0
    with fresh_kb() as kb:
        with Timer() as t:
            kb.ingest_folder(folder, "bench")
        ingest_s = t.s
        r1, r5, r10, right_scores = [], [], [], []
        with Timer() as t:
            for caption, filename in captions:
                hits = kb.search(caption, k=10, kind="image")
                names = [h["title"] for h in hits]
                r1.append(filename in names[:1])
                r5.append(filename in names[:5])
                r10.append(filename in names[:10])
                right_scores += [h["score"] for h in hits if h["title"] == filename]
        query_ms = 1000 * t.s / len(captions)
        claims = [
            q["text"] for q in load_dataset("BeIR/scifact", "queries", split="queries")
        ]
        unrelated = [kb.search(c, k=1, kind="image")[0]["score"] for c in claims[:300]]
    found = sum(r10)
    print(
        f"Flickr30k 1k: {len(captions) // 5} photos, {len(captions)} captions; "
        f"image ingest {ingest_s:.0f} s, {query_ms:.0f} ms/query"
    )
    print(
        table(
            [
                {
                    "recall@1": f"{mean(r1):.3f}",
                    "recall@5": f"{mean(r5):.3f}",
                    "recall@10": f"{mean(r10):.3f}",
                }
            ],
            ["recall@1", "recall@5", "recall@10"],
        )
    )
    below = sum(s < CUTOFF for s in right_scores)
    median_right = sorted(right_scores)[len(right_scores) // 2]
    print(
        f"\nCut-off {CUTOFF}: {below} of {found} correct photos found in the top 10 score below it "
        f"({below / found:.0%}); median correct score {median_right:.3f}"
    )
    over = sum(s >= CUTOFF for s in unrelated)
    median_unrelated = sorted(unrelated)[len(unrelated) // 2]
    print(
        f"Unrelated text: {over} of {len(unrelated)} SciFact claims would surface a photo "
        f"(best photo score >= {CUTOFF}); max {max(unrelated):.3f}, median {median_unrelated:.3f}"
    )


if __name__ == "__main__":
    main()
