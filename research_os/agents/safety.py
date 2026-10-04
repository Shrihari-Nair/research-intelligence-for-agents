"""Prompt-injection hygiene for UNTRUSTED fetched content (hard rule 7).

Detection is a deterministic first line of defence, not a guarantee: sources are also delimited and labelled as data in every
prompt, and the verifier judges every claim against its quote.
"""
from __future__ import annotations

import re

_PATTERNS = [re.compile(p, re.I) for p in (
    r"\b(ignore|disregard|forget)\b[^.]{0,40}\b(previous|prior|above|earlier|all)\b[^.]{0,30}\b(instructions?|prompts?|rules?|messages?)\b",
    r"\byou are now\b",
    r"\b(reveal|print|show|output|repeat)\b[^.]{0,30}\b(system prompt|api key|secret|password|credentials)\b",
    r"\bnew instructions?\s*:",
    r"\b(admin|developer|dan|jailbreak) mode\b",
)]
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*\s*")


def find_injection(text: str) -> list[str]:
    """Matched instruction-like snippets in `text` (empty list when clean)."""
    return [m.group(0) for p in _PATTERNS for m in [p.search(text or "")] if m]


def injection_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences in `text` that look like instructions to an AI."""
    return [(m.start(), m.end()) for m in _SENTENCE.finditer(text or "") if find_injection(m.group(0))]


def strip_injection(text: str) -> tuple[str, list[str]]:
    """Remove sentences that look like instructions to an AI. Returns (clean text, removed sentences)."""
    kept, removed = [], []
    for m in _SENTENCE.finditer(text):
        (removed if find_injection(m.group(0)) else kept).append(m.group(0))
    return "".join(kept).strip(), [r.strip() for r in removed]
