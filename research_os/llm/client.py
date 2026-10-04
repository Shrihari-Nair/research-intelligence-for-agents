"""LLM client interface + strict-JSON intent parsing shared by all implementations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

FALLBACK_INTENT = "GENERAL_QUERY"


@dataclass
class LLMResponse:
    """One LLM completion with the usage needed for budgets, cost and tracing."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


class LLMClient(Protocol):
    """Anything that can complete a prompt (research pipeline) or classify/phrase (legacy)."""

    def complete(self, prompt: str, *, model: str | None = None, json_mode: bool = False,
                 max_tokens: int = 1024, meta: dict | None = None) -> LLMResponse:
        """Return the raw completion plus usage. `meta` (e.g. {"prompt": "supervisor.v1"}) is logged, never sent."""

    def classify_intent(self, text: str, intents: list[str]) -> dict:
        """Return {"intent","reason","input_tokens","output_tokens","failed"}."""

    def generate(self, prompt: str) -> str:
        """Return free-form text (used by agents to phrase responses)."""


def build_classify_prompt(text: str, intents: list[str]) -> str:
    """Prompt asking for strict JSON with one allowed label."""
    return (
        "You route user utterances for a medical clinic voice assistant.\n"
        f"Allowed intents: {', '.join(intents)}.\n"
        "If the user may be in danger or describes a medical emergency, choose EMERGENCY.\n"
        'Reply with ONLY JSON: {"intent": "<one allowed intent>", "reason": "<short reason>"}.\n'
        f"Utterance: {text}"
    )


def parse_llm_intent(raw: str, intents: list[str]) -> dict:
    """Validate raw model output; on any problem return FALLBACK_INTENT with failed=True."""
    fail = {"intent": FALLBACK_INTENT, "reason": "unparseable or invalid LLM output", "failed": True}
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
        obj = json.loads(cleaned)
        if not isinstance(obj, dict) or obj.get("intent") not in intents:
            return fail
        return {"intent": obj["intent"], "reason": str(obj.get("reason", "")), "failed": False}
    except (ValueError, TypeError):
        return fail
