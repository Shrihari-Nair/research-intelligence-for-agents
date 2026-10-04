"""Baseline router: frozen pretrained MiniLM embeddings + LogisticRegression head."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from research_os.common import load_intents
from research_os.router.model import IntentRouter


class BaselineRouter(IntentRouter):
    """Same interface as IntentRouter, but no contrastive fine-tuning."""

    def __init__(self, encoder: SentenceTransformer, head, intents: list[str]):
        self.encoder, self.head, self.intents = encoder, head, intents

    @classmethod
    def load(cls, path: str, intents: list[str] | None = None) -> BaselineRouter:
        """Load encoder name + head from a directory written by baseline.py."""
        meta = joblib.load(Path(path) / "head.joblib")
        return cls(SentenceTransformer(meta["encoder"], device="cpu"), meta["head"],
                   intents or load_intents())

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        texts = [self._check(t) for t in texts]
        emb = self.encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
        probs = np.zeros((len(texts), len(self.intents)))
        probs[:, self.head.classes_] = self.head.predict_proba(emb)
        return self._records(probs)
