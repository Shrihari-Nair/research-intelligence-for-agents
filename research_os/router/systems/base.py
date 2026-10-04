"""Shared types and the emergency rule for all routing systems."""
from __future__ import annotations

from typing import Protocol, TypedDict

EMERGENCY = "EMERGENCY"


class RoutingRecord(TypedDict):
    """What every system returns for one utterance."""

    intent: str
    confidence: float
    source: str  # "minilm" | "llm" | "emergency_rule"
    llm_called: bool
    llm_failed: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int


class RoutingSystem(Protocol):
    """A pluggable intent-routing strategy."""

    def route(self, text: str) -> RoutingRecord:
        """Route one utterance."""


def emergency_triggered(pred: dict, floor: float) -> bool:
    """True if EMERGENCY is the top intent or its probability reaches the floor."""
    return pred["intent"] == EMERGENCY or pred["probs"].get(EMERGENCY, 0.0) >= floor
