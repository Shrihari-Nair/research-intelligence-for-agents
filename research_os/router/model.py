"""IntentRouter: thin wrapper over a saved SetFit model returning intent + confidence."""
from __future__ import annotations

import numpy as np
from setfit import SetFitModel

from research_os.common import load_intents


class IntentRouter:
    """Classifies an utterance into one intent with probabilities."""

    def __init__(self, model: SetFitModel, intents: list[str]):
        self.model = model
        self.intents = intents

    @classmethod
    def load(cls, path: str, intents: list[str] | None = None) -> IntentRouter:
        """Load a model saved with `save_pretrained`."""
        return cls(SetFitModel.from_pretrained(path), intents or load_intents())

    @staticmethod
    def _check(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        return text.strip()

    def _records(self, probs: np.ndarray) -> list[dict]:
        out = []
        for row in probs:
            row = np.asarray(row, dtype=float)
            row = row / row.sum()
            top = int(row.argmax())
            out.append({"intent": self.intents[top], "confidence": float(row[top]),
                        "probs": {n: float(p) for n, p in zip(self.intents, row, strict=True)}})
        return out

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Predict for many texts; same schema as `predict`."""
        texts = [self._check(t) for t in texts]
        probs = self.model.predict_proba(texts, batch_size=batch_size, as_numpy=True)
        return self._records(probs)

    def predict(self, text: str) -> dict:
        """Return {"intent", "confidence" (max prob), "probs" (dict over intents)}."""
        return self.predict_batch([text], batch_size=1)[0]
