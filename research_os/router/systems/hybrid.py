"""System C: emergency rule -> confidence threshold -> LLM fallback."""
from __future__ import annotations

import logging
import time

from research_os.common import load_intents
from research_os.llm.client import FALLBACK_INTENT
from research_os.router.systems.base import EMERGENCY, RoutingRecord, emergency_triggered

log = logging.getLogger(__name__)
HEALTHCARE_INTENTS = "configs/healthcare/intents.yaml"  # these systems encode healthcare semantics (EMERGENCY, GENERAL_QUERY)


class HybridSystem:
    """High confidence -> MiniLM; low confidence -> LLM; emergencies never wait on either."""

    def __init__(self, router, llm, cfg: dict, intents: list[str] | None = None):
        if cfg.get("threshold") is None:
            raise ValueError("routing threshold is not set; run `python -m research_os.router.calibration`")
        self.router, self.llm = router, llm
        self.threshold, self.floor = float(cfg["threshold"]), float(cfg["emergency_floor"])
        self.intents = intents or load_intents(HEALTHCARE_INTENTS)

    def route(self, text: str) -> RoutingRecord:
        t0 = time.perf_counter()
        pred = self.router.predict(text)
        rec: RoutingRecord = {"intent": pred["intent"], "confidence": pred["confidence"],
                              "source": "minilm", "llm_called": False, "llm_failed": False,
                              "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0}
        if emergency_triggered(pred, self.floor):
            rec["intent"], rec["source"] = EMERGENCY, "emergency_rule"
        elif pred["confidence"] < self.threshold:
            rec["llm_called"], rec["source"] = True, "llm"
            try:
                out = self.llm.classify_intent(text, self.intents)
            except Exception as e:  # LLM must never take routing down
                log.warning("LLM fallback raised: %s", e)
                out = {"intent": FALLBACK_INTENT, "failed": True}
            rec["intent"] = out["intent"] if out["intent"] in self.intents else FALLBACK_INTENT
            rec["llm_failed"] = bool(out.get("failed", False))
            rec["input_tokens"] = int(out.get("input_tokens", 0))
            rec["output_tokens"] = int(out.get("output_tokens", 0))
        rec["latency_ms"] = (time.perf_counter() - t0) * 1000
        return rec
