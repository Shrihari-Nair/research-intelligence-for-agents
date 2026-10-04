"""ResearchRouter: SetFit intent + deterministic entities -> RouterOutput (the pipeline's control point)."""
from __future__ import annotations

import time

from research_os.common import load_config
from research_os.contracts.router import RouterOutput, RoutingMode
from research_os.router.entities import EntityExtractor
from research_os.router.model import IntentRouter
from research_os.router.workflows import workflow_map


class ResearchRouter:
    """Wraps an IntentRouter (anything with .predict(text) -> {intent, confidence, probs})."""

    def __init__(self, intent_router, entities: EntityExtractor, threshold: float | None, workflows: dict[str, str]):
        if threshold is None:
            raise ValueError("routing threshold is not set; run `python -m research_os.router.calibration`")
        self.intent_router, self.entities = intent_router, entities
        self.threshold, self.workflows = float(threshold), workflows

    @classmethod
    def from_config(cls, model_path: str | None = None, training_cfg: str = "configs/training.yaml",
                    routing_cfg: str = "configs/routing.yaml") -> ResearchRouter:
        """Load the saved router, gazetteer, threshold and workflow map from config."""
        tcfg, rcfg = load_config(training_cfg), load_config(routing_cfg)
        router = IntentRouter.load(model_path or tcfg["paths"]["model"])
        return cls(router, EntityExtractor.from_config(), rcfg["threshold"], workflow_map())

    def route(self, text: str) -> RouterOutput:
        """Classify, extract entities, decide DIRECT vs LLM_FALLBACK. Raises ValueError on empty text."""
        t0 = time.perf_counter()
        pred = self.intent_router.predict(text)
        entities = self.entities.extract(text)
        mode = RoutingMode.DIRECT if pred["confidence"] >= self.threshold else RoutingMode.LLM_FALLBACK
        return RouterOutput(intent=pred["intent"], confidence=pred["confidence"], routing_mode=mode,
                            workflow=self.workflows[pred["intent"]], entities=entities,
                            latency_ms=(time.perf_counter() - t0) * 1000)
