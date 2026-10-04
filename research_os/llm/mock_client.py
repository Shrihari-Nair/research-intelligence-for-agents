"""Deterministic offline LLM stub (keyword rules) for tests and no-key runs."""
from __future__ import annotations

import time

from research_os.llm.client import FALLBACK_INTENT, LLMResponse

DEFAULT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("EMERGENCY", ("chest pain", "can't breathe", "cant breathe", "not breathing", "overdose",
                   "unconscious", "stroke", "heart attack", "bleeding", "ambulance", "collapsed")),
    ("CANCEL_APPOINTMENT", ("cancel", "call off", "remove my appointment", "drop my")),
    ("RESCHEDULE_APPOINTMENT", ("reschedule", "move my", "postpone", "push my", "shift my",
                                "change my appointment", "different slot")),
    ("BOOK_APPOINTMENT", ("book", "schedule", "see dr", "appointment with", "make an appointment")),
    ("FOLLOWUP_QUERY", ("follow up", "follow-up", "next visit", "come back", "recheck", "review")),
    ("PRESCRIPTION_QUERY", ("prescri", "dosage", "tablet", "medicine", "pills", "medication")),
    ("MEDICAL_RESEARCH", ("research", "study", "studies", "latest", "trial", "treatment for",
                          "pubmed", "evidence")),
]


class MockLLMClient:
    """Keyword-rule classifier. Token counts are fake and flagged with mock=True."""

    def __init__(self, delay_s: float = 0.0, rules=None):
        self.delay_s = delay_s
        self.rules = rules or DEFAULT_RULES
        self.calls = 0

    def classify_intent(self, text: str, intents: list[str]) -> dict:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        low = text.lower()
        intent = FALLBACK_INTENT
        for name, kws in self.rules:
            if name in intents and any(k in low for k in kws):
                intent = name
                break
        return {"intent": intent, "reason": "mock keyword rule", "failed": False,
                "input_tokens": len(text.split()) + 40, "output_tokens": 20, "mock": True}

    def generate(self, prompt: str) -> str:
        """Echo the text after 'Draft:' (so agents keep their own deterministic wording)."""
        return prompt.split("Draft:", 1)[-1].strip()


class ScriptedLLM:
    """Deterministic test LLM: returns queued responses in order, recording every prompt."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.metas: list[dict | None] = []
        self.calls = 0

    def complete(self, prompt: str, *, model: str | None = None, json_mode: bool = False,
                 max_tokens: int = 1024, meta: dict | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        self.metas.append(meta)
        self.calls += 1
        if not self.responses:
            raise RuntimeError("ScriptedLLM exhausted: more calls than scripted responses")
        text = self.responses.pop(0)
        return LLMResponse(text=text, input_tokens=len(prompt.split()), output_tokens=len(text.split()),
                           model=model or "scripted")
