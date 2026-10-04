"""Deterministic entity extraction for routed requests. No LLM, no model: gazetteer + regex only.

Returns {"technologies": [...], "companies": [...], "topics": [...], "time_scope": [...]} (all lists of str,
canonical names for gazetteer hits, original text for quoted/derived topics).
"""
from __future__ import annotations

import re

from research_os.common import load_config

MAX_CHARS = 5000
_CAMEL = re.compile(r"\b(?=\w*[a-z])(?=\w*[A-Z])[A-Za-z][A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*\b|\b\w+\.(?:js|py|io)\b")
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,80})[\"“”]|(?<!\w)'([^']{3,60})'(?!\w)")
_TOPIC = re.compile(
    r"\b(?:on|about|regarding|around|covering)\s+(?P<t>[a-z0-9][\w\s\-/+&']{2,80}?)"
    r"(?=\s+(?:from|since|over|for|that|which|with|please|thanks|and then|in the|in 19|in 20)\b|[,.;?!]|$)", re.I)
_TIME = [
    re.compile(r"\b(latest|recent(?:ly)?|newest|lately|this (?:year|month|week)|today|these days)\b", re.I),
    re.compile(r"\bsince (?:19|20)\d{2}\b", re.I),
    re.compile(r"\bin (?:19|20)\d{2}\b", re.I),
    re.compile(r"\bbetween (?:19|20)\d{2} and (?:19|20)\d{2}\b", re.I),
    re.compile(r"\b(?:last|past) (?:\d+|one|two|three|four|five|six|a|few) (?:days?|weeks?|months?|years?)\b", re.I),
    re.compile(r"\b(?:last|past) (?:year|month|week|decade)\b", re.I),
]
_LEAD = re.compile(r"^(?:the|a|an|some|any)\s+", re.I)
_TRAIL_TIME = re.compile(r"\s+(?:latest|recent|recently|lately|today)$", re.I)


class EntityExtractor:
    """Gazetteer + regex entity extraction."""

    def __init__(self, gazetteer: dict):
        self._patterns: list[tuple[re.Pattern, str, str]] = []  # (regex, category, canonical)
        for cat in ("technologies", "companies"):
            for entry in gazetteer.get(cat, []):
                name = entry["name"]
                flags = 0 if entry.get("exact_case") else re.IGNORECASE
                self._patterns.append((self._rx(name, flags), cat, name))
                for alias in entry.get("aliases", []):
                    self._patterns.append((self._rx(alias, re.IGNORECASE), cat, name))

    @classmethod
    def from_config(cls, path: str = "configs/gazetteer.yaml") -> EntityExtractor:
        """Build from the YAML gazetteer."""
        return cls(load_config(path))

    @staticmethod
    def _rx(term: str, flags: int) -> re.Pattern:
        return re.compile(rf"(?<![\w]){re.escape(term)}(?![\w])", flags)

    def extract(self, text: str) -> dict[str, list[str]]:
        """Extract entities from one request; never raises on odd input."""
        text = (text or "")[:MAX_CHARS]
        hits: list[tuple[int, str, str]] = []
        covered: list[tuple[int, int]] = []
        for rx, cat, name in self._patterns:
            for m in rx.finditer(text):
                if any(s <= m.start() < e or s < m.end() <= e for s, e in covered):
                    continue
                covered.append((m.start(), m.end()))
                hits.append((m.start(), cat, name))
        for m in _CAMEL.finditer(text):
            if not any(s <= m.start() < e for s, e in covered):
                hits.append((m.start(), "technologies", m.group(0)))
        out: dict[str, list[str]] = {"technologies": [], "companies": [], "topics": [], "time_scope": []}
        seen: set[tuple[str, str]] = set()
        for _, cat, name in sorted(hits):
            if (cat, name.lower()) not in seen:
                seen.add((cat, name.lower()))
                out[cat].append(name)
        known = {n.lower() for c in ("technologies", "companies") for n in out[c]}
        for m in _QUOTED.finditer(text):
            self._add_topic(out, (m.group(1) or m.group(2)).strip(), known)
        m = _TOPIC.search(text)
        if m:
            t = _TRAIL_TIME.sub("", _LEAD.sub("", m.group("t").strip()))
            self._add_topic(out, t, known)
        for rx in _TIME:
            for m in rx.finditer(text):
                tag = re.sub(r"\s+", " ", m.group(0).lower())
                if tag not in out["time_scope"]:
                    out["time_scope"].append(tag)
        return out

    @staticmethod
    def _add_topic(out: dict, topic: str, known: set[str]) -> None:
        if topic and topic.lower() not in known and topic not in out["topics"]:
            out["topics"].append(topic)
