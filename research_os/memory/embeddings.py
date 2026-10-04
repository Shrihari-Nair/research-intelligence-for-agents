"""Sentence embeddings for memory search. One shared MiniLM-family encoder (no vector database; brute-force cosine)."""
from __future__ import annotations

import numpy as np

_CACHE: dict[str, object] = {}


class SentenceEmbedder:
    """Lazy-loaded SentenceTransformer shared across the process. Returns L2-normalized float32 vectors."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def _model(self):
        if self.model_name not in _CACHE:
            from sentence_transformers import SentenceTransformer
            _CACHE[self.model_name] = SentenceTransformer(self.model_name, device="cpu")
        return _CACHE[self.model_name]

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests/CI (no model download). Not for production search quality."""

    dim = 128

    def encode(self, texts: list[str]) -> np.ndarray:
        import re
        import zlib

        from research_os.agents.text import STOP
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w not in STOP:
                    v[zlib.crc32(w.encode()) % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.vstack(out)
