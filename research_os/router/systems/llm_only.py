"""System A: every utterance goes to the LLM for intent classification."""
from __future__ import annotations

import time

from research_os.common import load_intents
from research_os.llm.client import FALLBACK_INTENT
from research_os.router.systems.base import RoutingRecord
from research_os.router.systems.hybrid import HEALTHCARE_INTENTS


class LLMOnlySystem:
    """No MiniLM, no emergency rule: the LLM alone decides the intent."""

    def __init__(self, llm, cfg: dict | None = None, intents: list[str] | None = None):
        self.llm, self.intents = llm, intents or load_intents(HEALTHCARE_INTENTS)

    def route(self, text: str) -> RoutingRecord:
        t0 = time.perf_counter()
        try:
            out = self.llm.classify_intent(text, self.intents)
        except Exception:
            out = {"intent": FALLBACK_INTENT, "failed": True}
        intent = out["intent"] if out["intent"] in self.intents else FALLBACK_INTENT
        return {"intent": intent, "confidence": 1.0, "source": "llm", "llm_called": True,
                "llm_failed": bool(out.get("failed", False)),
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": int(out.get("input_tokens", 0)),
                "output_tokens": int(out.get("output_tokens", 0))}
