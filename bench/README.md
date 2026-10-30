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

## Which local model?

`models.py` runs the text benchmarks (SciFact, ViDoRe PDF) across embedding models and the
image benchmark across CLIP sizes, and `charts.py` draws the charts below from
`results/models.json`.

```bash
uv run python models.py                                          # all models; resumes if stopped
LIVE_EMBED_URL=http://127.0.0.1:8081 uv run python models.py     # also a llama.cpp server's model
uv run python charts.py
```

How it is run, so the numbers compare fairly:

- Each model uses the query and document prompts from its model card (e5: `query:` and
  `passage:`, bge and arctic: the "Represent this sentence" query prefix, Qwen3: its query
  instruction, MiniLM: none).
- Inputs are capped at 512 tokens, the limit most of these models already have.
- Quality is measured on the GPU; speed is measured on the CPU in float32, one question at a
  time, which is how a laptop would run it. The time per question is the embedding plus the
  search.
- Qwen3-Embedding-8B is too big for a laptop. It was measured through a llama.cpp server
  (Q6_K GGUF, last-token pooling, output cut to 1536 dimensions), so its time is a GPU time.
- nomic-embed-text-v1.5 is left out: its custom model code does not run on current
  `transformers`.

![Text search quality vs model size](results/models-scifact.svg)

Keyword search alone scores 0.676. Hybrid search lifts every small model and narrows their
spread from 0.645 to 0.740 down to 0.707 to 0.752. Qwen3-Embedding-8B reaches 0.796 with vector
search alone, and is the one model that keyword search makes worse (0.775 hybrid).

![Finding the right PDF page](results/models-vidore.svg)

![Time per question on a CPU](results/models-cpu.svg)

![Finding a photo from a sentence](results/models-clip.svg)

## A model before or after the search?

Two common add-ons, measured on the same benchmarks with bge-small hybrid search as the base.
Numbers are SciFact nDCG@10, the share of ViDoRe questions with the right PDF page first, and
with it in the top five.

**After: reranking** (`rerank.py`). A cross-encoder reads the question with each of the top 50
candidates and reorders them. Time is per question, on the CPU, for those 50 pairs.

| | SciFact | PDF page first | PDF top 5 | CPU time |
|---|---|---|---|---|
| no reranker | 0.723 | 74% | 95% | |
| ms-marco-MiniLM-L6-v2 (22M) | 0.696 | 76% | 92% | 1.2 s |
| bge-reranker-base (278M) | 0.721 | 72% | 94% | 8.2 s |

Neither helps here, and both cost far more than the 12 ms search. ms-marco is trained on web
search queries, which is a likely reason it does worse on scientific claims.

**Before: a small language model** (`presearch.py`). *Rewrite* asks the model for a short
keyword query. *HyDE* asks it for a two- or three-sentence answer and searches with the
question plus that answer, since a made-up answer reads more like the documents.

| | SciFact | PDF page first | PDF top 5 | added time |
|---|---|---|---|---|
| nothing | 0.722 | 74% | 95% | |
| Qwen2.5-0.5B, rewrite | 0.670 | 71% | 93% | 81 ms (GPU) |
| Qwen2.5-0.5B, HyDE | 0.708 | 77% | 97% | 150 ms (GPU) |
| Qwen2.5-1.5B, rewrite | 0.694 | 74% | 92% | 1.8 s (CPU) |
| Qwen2.5-1.5B, HyDE | 0.732 | 77% | 96% | 4.1 s (CPU) |

Rewriting hurts. HyDE with the 1.5B model helps on both benchmarks, by about a point on
SciFact and three questions on the PDF, at the cost of a model call before every search.
The 1.5B CPU timings were taken while other jobs shared the machine, so read them as an upper
bound.
