"""Embedding models that fit on a laptop.

The default is BAAI/bge-small-en-v1.5 through sentence-transformers: about
130 MB, 384 dimensions, fast on a CPU. It runs inside the server process, so
there is nothing else to install or keep running.

Set LANCE_KB_MODEL to another sentence-transformers model, or to "hash" for a
tiny deterministic embedder that needs no download (tests and CI).
"""

import hashlib
import logging
import math
import os
import re
from functools import cache

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models retrieve better when the query, not the document, carries this prefix.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "
}


class HashEmbedder:
    """Bag-of-words hashing: no model, deterministic, good enough for tests."""

    name = "hash"
    dim = 256

    def encode(self, texts, query=False):
        vectors = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class SentenceTransformerEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer

        # The model download logs every HTTP request; on stdio they only clutter the MCP log.
        for logger in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
            logging.getLogger(logger).setLevel(logging.WARNING)
        self.name = name
        self.model = SentenceTransformer(
            name, device=os.environ.get("LANCE_KB_DEVICE") or None
        )
        self.dim = self.model.get_embedding_dimension()
        self.prefix = QUERY_PREFIXES.get(name, "")

    def encode(self, texts, query=False):
        if query and self.prefix:
            texts = [self.prefix + t for t in texts]
        return self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        ).tolist()


@cache
def get_embedder(name=None):
    name = name or os.environ.get("LANCE_KB_MODEL", DEFAULT_MODEL)
    return HashEmbedder() if name == "hash" else SentenceTransformerEmbedder(name)
