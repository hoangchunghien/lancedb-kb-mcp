# Benchmarks

Three public datasets, one per thing the knowledge base claims to do: find text, find the
right PDF page, and find a picture from a sentence. Each script downloads its data on first
run (into `bench/.cache/`, not committed), builds a fresh knowledge base in a temp folder,
and prints a table.

```bash
uv sync --group bench
cd bench
uv run python scifact.py        # text: vector vs keyword vs hybrid
uv run python vidore_pdf.py     # PDF: does the right page come back?
uv run python flickr_images.py  # images: CLIP text-to-image, and the 0.28 cut-off
```

Results below are from 30 Oct 2026 with the default models (bge-small-en-v1.5 for text,
CLIP ViT-B/32 for images), on an RTX 4070 SUPER. Timings on a CPU are slower; the scores
are the same.

## Text: BEIR SciFact

5,183 scientific abstracts and the 300 test claims with relevance labels
([BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact), CC BY-SA 4.0).

| mode | nDCG@10 | recall@10 | ms/query |
|---|---|---|---|
| vector | 0.713 | 0.836 | 10 |
| text (BM25) | 0.676 | 0.804 | 8 |
| hybrid | 0.722 | 0.846 | 14 |

The keyword-only score is close to the 0.665 BM25 baseline in the BEIR paper, which is a
useful check that the setup is sound. Hybrid is the best of the three here, by a small
margin over vector search.

## PDF: ViDoRe V3, computer science

[ViDoRe V3](https://huggingface.co/datasets/vidore/vidore_v3_computer_science) pairs
questions with the textbook pages that answer them, checked by people. The script ingests
the real PDF of OpenStax *Introduction to Computer Science* through the normal PDF path, and
a question counts as answered at k when a labelled page is among the top k pages.

| corpus | nDCG@10 | hit@1 | hit@5 | hit@10 | ms/query |
|---|---|---|---|---|---|
| the PDF, through pypdf (960 chunks) | 0.680 | 0.74 | 0.95 | 0.97 | 10 |
| ViDoRe's per-page Markdown (944 pages) | 0.691 | 0.77 | 0.96 | 0.97 | 12 |

Ingesting the 945-page PDF took 8 s. Plain pypdf text costs very little against
ViDoRe's cleaner Markdown on this book.

Two things to know about this run:

- The labels belong to a specific edition. OpenStax has since re-issued both textbooks
  with different page breaks, so the script downloads the December 2024 copy of the
  Computer Science book from the Internet Archive, which has the same 945 pages (893 of
  934 text pages match ViDoRe's page for page). The 945-page edition of the Python
  textbook is no longer available, so its 79 questions are left out: 132 of the 215
  English questions are used.
- OpenStax lists both books as CC BY-NC-SA. The PDF is downloaded at run time and never
  committed.

## Images: Flickr30k 1k test

1,000 photos with five captions each, the standard test split for text-to-image retrieval
([nlphuji/flickr_1k_test_image_text_retrieval](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval),
Flickr terms, research use). Every caption is a query.

| recall@1 | recall@5 | recall@10 |
|---|---|---|
| 0.594 | 0.845 | 0.902 |

Ingesting the 1,000 photos took 41 s; a search took 44 ms.

The 0.28 cut-off (`LANCE_KB_IMAGE_MIN_SCORE`) keeps photos out of results that are not
about them. Measured both ways:

- It hides 10% of the correct photos (429 of the 4,511 found in the top 10 score below
  it). The median correct score is 0.317.
- It still lets some unrelated text through: for 26 of 300 SciFact claims (9%), the best
  of the 1,000 photos scores 0.28 or more. The highest was 0.315.

So the cut-off is a trade-off rather than a clean line. CLIP scores for matching and
unrelated pairs overlap. Raise it if unrelated photos bother you, lower it if photos go
missing.
