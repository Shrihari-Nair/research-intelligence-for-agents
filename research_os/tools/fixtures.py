"""Fixture mode: canned, deterministic corpus of Sources for tests, CI and evaluation replay seeds."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from research_os.contracts import FailureType, Source, canonical_url, source_id_for_url
from research_os.tools.registry import ToolError

_TOKEN = re.compile(r"[a-z0-9]+")


class FixtureCorpus:
    """In-memory corpus loaded from <dir>/corpus.json (SYNTHETIC documents)."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self._by_url = {canonical_url(d["url"]): d for d in docs}

    @classmethod
    def load(cls, directory: str) -> FixtureCorpus:
        """Read <directory>/corpus.json."""
        data = json.loads((Path(directory) / "corpus.json").read_text())
        return cls(data["documents"])

    @staticmethod
    def _source(d: dict, tool: str | None = None) -> Source:
        return Source(source_id=source_id_for_url(d["url"]), title=d["title"], url=canonical_url(d["url"]),
                      source_type=d["source_type"], published_at=datetime.fromisoformat(d["published_at"]),
                      retrieved_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"), content=d["content"],
                      tool=tool or d["tool"])

    def search(self, query: str, tool: str, max_results: int) -> list[Source]:
        """Top documents for this tool by token overlap (ties broken by source id, so results are deterministic)."""
        q = set(_TOKEN.findall(query.lower()))
        scored = []
        for d in self.docs:
            if d["tool"] != tool:
                continue
            haystack = set(d["tags"]) | set(_TOKEN.findall(d["title"].lower())) | set(_TOKEN.findall(d["content"].lower()))
            tag_hits = len(q & set(d["tags"]))
            score = tag_hits * 3 + len(q & haystack)
            if score > 0 and q & haystack:
                scored.append((-score, source_id_for_url(d["url"]), d))
        return [self._source(d) for _, _, d in sorted(scored, key=lambda t: (t[0], t[1]))[:max_results]]

    def fetch(self, url: str) -> Source:
        """The corpus document at `url` (canonicalized) or ToolError(INVALID_SOURCE)."""
        d = self._by_url.get(canonical_url(url))
        if d is None:
            raise ToolError(FailureType.INVALID_SOURCE, f"url not available in fixtures: {url}")
        return self._source(d, tool="fetch_url")
