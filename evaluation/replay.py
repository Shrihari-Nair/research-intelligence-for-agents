"""Replay infrastructure: frozen per-task source corpus (no web drift) and a content-addressed LLM cache (identical reruns)."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

from research_os.common import load_config
from research_os.contracts import FailureType, Source, canonical_url
from research_os.llm.client import LLMResponse
from research_os.tools import build_registry
from research_os.tools.registry import ToolError, ToolRegistry

_TOKEN = re.compile(r"[a-z0-9]+")
_TOOL_TYPES = {"arxiv_search": {"paper"}, "pubmed_search": {"paper"}, "github_search": {"repository"}}


class CassetteCorpus:
    """All sources recorded live for one task. Serves search/fetch for ANY system, so systems that issue different queries are
    compared on the same frozen evidence."""

    def __init__(self, sources: list[Source]):
        self.sources = sources
        self._by_url = {canonical_url(s.url): s for s in sources}

    @classmethod
    def load(cls, task_id: str, directory: str) -> CassetteCorpus:
        raw = json.loads((Path(directory) / f"{task_id}.sources.json").read_text())
        return cls([Source(**r) for r in raw])

    def search(self, query: str, tool: str, max_results: int) -> list[Source]:
        """Top sources by token overlap. Scholarly tools return papers, github_search repositories, web_search anything."""
        q = set(_TOKEN.findall(query.lower()))
        want = _TOOL_TYPES.get(tool)
        scored = []
        for s in self.sources:
            if want is not None and s.source_type not in want:
                continue
            hay = set(_TOKEN.findall(f"{s.title} {s.content}".lower()))
            overlap = len(q & hay)
            if overlap:
                scored.append((-overlap, s.source_id, s))
        return [s.model_copy(update={"tool": tool}) for _, _, s in sorted(scored, key=lambda t: (t[0], t[1]))[:max_results]]

    def fetch(self, url: str) -> Source:
        s = self._by_url.get(canonical_url(url))
        if s is None:
            raise ToolError(FailureType.INVALID_SOURCE, f"url not in the frozen corpus: {url}")
        return s.model_copy(update={"tool": "fetch_url"})


def build_replay_registry(corpus: CassetteCorpus, cfg: dict | None = None) -> ToolRegistry:
    """The normal enforced registry (schemas, permissions, budgets, policies) backed by the frozen corpus."""
    cfg = cfg or load_config("configs/tools.yaml")
    return build_registry(cfg, mode="fixture", corpus=corpus)


class CacheMiss(RuntimeError):
    """Offline replay needed an LLM response that was never cached."""


class CachedLLM:
    """LLM client with a persistent content-addressed cache. `base=None` means offline: a miss raises CacheMiss."""

    def __init__(self, base, cache_path: str):
        self.base, self.path = base, Path(cache_path)
        self._lock = threading.Lock()
        self._mem: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._mem[e["key"]] = e["response"]

    @staticmethod
    def key(prompt: str, model: str | None, json_mode: bool, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([model or "default", json_mode, max_tokens, prompt]).encode()).hexdigest()

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}

    def complete(self, prompt: str, *, model: str | None = None, json_mode: bool = False, max_tokens: int = 1024,
                 meta: dict | None = None) -> LLMResponse:
        k = self.key(prompt, model, json_mode, max_tokens)
        with self._lock:
            if k in self._mem:
                self.hits += 1
                return LLMResponse(**self._mem[k])
        if self.base is None:
            raise CacheMiss(f"no cached LLM response for model={model!r} (offline replay); re-run with a live LLM to populate the cache")
        resp = self.base.complete(prompt, model=model, json_mode=json_mode, max_tokens=max_tokens, meta=meta)
        data = {"text": resp.text, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "model": resp.model,
                "latency_ms": resp.latency_ms}
        with self._lock:
            self.misses += 1
            self._mem[k] = data
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": k, "response": data}) + "\n")
        return resp
