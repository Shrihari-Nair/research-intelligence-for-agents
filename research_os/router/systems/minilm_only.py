"""System B: MiniLM intent goes straight to the agent (emergency rule still applies)."""
from __future__ import annotations

import time

from research_os.router.systems.base import EMERGENCY, RoutingRecord, emergency_triggered


class MiniLMOnlySystem:
    """No LLM fallback."""

    def __init__(self, router, cfg: dict):
        self.router, self.floor = router, cfg["emergency_floor"]

    def route(self, text: str) -> RoutingRecord:
        t0 = time.perf_counter()
        pred = self.router.predict(text)
        em = emergency_triggered(pred, self.floor)
        return {"intent": EMERGENCY if em else pred["intent"], "confidence": pred["confidence"],
                "source": "emergency_rule" if em else "minilm", "llm_called": False,
                "llm_failed": False, "latency_ms": (time.perf_counter() - t0) * 1000,
                "input_tokens": 0, "output_tokens": 0}
