"""Embedding models that fit on a laptop.

The default is BAAI/bge-small-en-v1.5 through sentence-transformers: about
130 MB, 384 dimensions, fast on a CPU. It runs inside the server process, so
there is nothing else to install or keep running.

Images use CLIP (clip-ViT-B-32, about 600 MB, 512 dimensions), which puts a
picture and a sentence describing it close together, so a text question can
find an image. It is loaded only when the knowledge base has images.

Set LANCE_KB_MODEL to another sentence-transformers model, or to "hash" for a
tiny deterministic embedder that needs no download (tests and CI); "hash" also
swaps CLIP for a hashing stand-in. LANCE_KB_IMAGE_MODEL overrides the image model.
"""

import hashlib
import logging
import math
import os
import re
from functools import cache
from pathlib import Path

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"
# (query prefix, document prefix) from each model card, for models whose config does not
# carry its own prompts. Models that do (Snowflake arctic-embed, mxbai, Qwen3-Embedding)
# use theirs.
_BGE = "Represent this sentence for searching relevant passages: "
PREFIXES = {
    "BAAI/bge-small-en-v1.5": (_BGE, ""),
    "BAAI/bge-base-en-v1.5": (_BGE, ""),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
}


def _hash_vector(text, dim):
    v = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class HashEmbedder:
    """Bag-of-words hashing: no model, deterministic, good enough for tests."""

    name = "hash"
    dim = 256

    def encode(self, texts, query=False):
        return [_hash_vector(text, self.dim) for text in texts]


class HashImageEmbedder:
    """Test stand-in for CLIP: an image is embedded by the words in its file name."""

    name = "hash"
    dim = 64

    def encode_images(self, paths):
        return [_hash_vector(Path(p).stem.replace("-", " "), self.dim) for p in paths]

    def encode_text(self, texts):
        return [_hash_vector(text, self.dim) for text in texts]


class SentenceTransformerEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        # The model download logs every HTTP request; on stdio they only clutter the MCP log.
        for logger in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
            logging.getLogger(logger).setLevel(logging.WARNING)
        self.name = name
        device = os.environ.get("LANCE_KB_DEVICE") or None
        # Some models default to half precision, which is slow on a CPU.
        kwargs = {"model_kwargs": {"dtype": "float32"}} if device == "cpu" else {}
        self.model = SentenceTransformer(
            name,
            device=device,
            trust_remote_code=name.startswith("nomic-ai/"),
            **kwargs,
        )
        self.dim = self.model.get_embedding_dimension()
        prompts = self.model.prompts or {}
        self.prefix, self.doc_prefix = PREFIXES.get(
            name, (prompts.get("query", ""), prompts.get("document", ""))
        )

    def encode(self, texts, query=False):
        prefix = self.prefix if query else self.doc_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        return self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        ).tolist()


class ClipEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(
            name, device=os.environ.get("LANCE_KB_DEVICE") or None
        )
        self.dim = self.model.get_embedding_dimension()

    def _encode(self, items):
        return self.model.encode(
            items, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def encode_images(self, paths):
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return self._encode(images)

    def encode_text(self, texts):
        return self._encode(list(texts))


@cache
def get_image_embedder(name=None):
    if os.environ.get("LANCE_KB_MODEL") == "hash":
        return HashImageEmbedder()
    return ClipEmbedder(
        name or os.environ.get("LANCE_KB_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    )


@cache
def get_embedder(name=None):
    name = name or os.environ.get("LANCE_KB_MODEL", DEFAULT_MODEL)
    return HashEmbedder() if name == "hash" else SentenceTransformerEmbedder(name)
