"""Offline DEMO LLM: deterministic, extractive stand-in so the API/UI/CI can run the full pipeline without a key.

It is NOT a language model: it picks sentences. Never use it for evaluation numbers (runs record llm_backend='demo').
"""
from __future__ import annotations

import json
import re

from research_os.llm.client import LLMResponse

_PASSAGE = re.compile(r"<<<SOURCE id=(\S+)\n(.*?)\nSOURCE>>>", re.S)
_CLAIM = re.compile(r"<<<CLAIM id=(\S+)\n", re.S)
_LISTED = re.compile(r"^(c\d+): (.+)$", re.M)
_SENT = re.compile(r"[^.!?\n]{25,240}[.!?]")


class DemoLLM:
    """Pattern-driven responses for each prompt in research_os/prompts."""

    BACKEND = "demo"

    def __init__(self) -> None:
        self.calls = 0

    def _respond(self, prompt: str) -> dict:
        if "You are the supervisor of a research system" in prompt:
            return {"workflow": "general_research", "rationale": "demo supervisor defaults to general research"}
        if "You extract verifiable claims" in prompt:
            claims, seen = [], set()
            for sid, text in _PASSAGE.findall(prompt):
                taken = 0
                for m in _SENT.finditer(text):
                    sentence = m.group(0).strip()
                    if sentence not in seen and len(claims) < 12 and taken < 3:
                        seen.add(sentence)
                        taken += 1
                        claims.append({"claim": sentence, "source_id": sid, "evidence_quote": sentence, "confidence": 0.6})
            return {"claims": claims}
        if "strict fact-checking judge" in prompt:
            return {"judgements": [{"claim_id": cid, "verdict": "SUPPORTED", "rationale": "demo judge: quote is the claim"}
                                   for cid in _CLAIM.findall(prompt)]}
        if "genuinely contradict each other" in prompt:
            return {"checks": [{"id": cid, "contradict": True, "reason": "demo: confirm every candidate"}
                               for cid in re.findall(r"<<<CANDIDATE id=(k\d+)", prompt)]}
        if "You write the findings section" in prompt:
            listed = _LISTED.findall(prompt)
            return {"summary": f"{len(listed)} verified statement(s) were found for this request (demo summary).",
                    "findings": [{"statement": text, "claim_ids": [cid]} for cid, text in listed]}
        if "You are the planner of a research system" in prompt:
            return {"extra_steps": []}
        return {}

    def complete(self, prompt: str, *, model: str | None = None, json_mode: bool = False, max_tokens: int = 1024,
                 meta: dict | None = None) -> LLMResponse:
        self.calls += 1
        text = json.dumps(self._respond(prompt))
        return LLMResponse(text=text, input_tokens=len(prompt.split()), output_tokens=len(text.split()), model=model or "demo")
