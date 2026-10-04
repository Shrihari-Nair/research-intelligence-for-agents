"""Source Analysis Agent: read sources, extract query-relevant passages with offsets. Deterministic (no LLM)."""
from __future__ import annotations

from research_os.agents.base import AgentFailure, agent_cfg, merged
from research_os.agents.safety import injection_spans
from research_os.agents.text import chunk_passages, keywords
from research_os.contracts import FailureType, ResearchStep
from research_os.tools.registry import ToolError


class SourceAgent:
    """Optionally upgrades snippets to full pages via fetch_url, then selects the best passages per source."""

    name = "source_agent"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or agent_cfg())["source"]

    def _clean_segments(self, content: str) -> tuple[list[tuple[int, int]], list[str]]:
        """Offsets of text outside injected sentences, plus the injected snippets that were excluded."""
        spans = injection_spans(content)
        segs, pos = [], 0
        for s, e in spans:
            if s > pos:
                segs.append((pos, s))
            pos = e
        if pos < len(content):
            segs.append((pos, len(content)))
        return segs, [content[s:e].strip() for s, e in spans]

    def run(self, step: ResearchStep, inputs: dict, ctx, request: str) -> dict:
        ids = merged(inputs, "source_ids")
        fetch_errors: list[dict] = []
        if "fetch_url" in step.tools:
            for sid in ids[: self.cfg["fetch_top"]]:
                src = ctx.sources.get(sid)
                if src is None or len(src.content) >= self.cfg["min_snippet_chars_for_fetch"]:
                    continue
                try:
                    ctx.tools.call("fetch_url", {"url": src.url}, ctx, agent=self.name)  # the store keeps the longer text
                except ToolError as e:
                    fetch_errors.append({"source_id": sid, "failure_type": e.failure_type.value, "error": str(e)})
        qwords = keywords(" ".join([step.objective, step.params.get("entity") or "", step.params.get("request") or request]), 10)
        passages, flagged = [], []
        for sid in ids:
            src = ctx.sources.get(sid)
            if src is None:
                continue
            segs, removed = self._clean_segments(src.content)
            if removed:
                flagged.append({"source_id": sid, "snippets": removed})
                ctx.events.emit(f"step:{step.id}", "info", {"possible prompt injection excluded": sid, "count": len(removed)})
            scored = []
            for s, e in segs:
                for ch in chunk_passages(src.content[s:e], self.cfg["passage_chars"]):
                    start, end = s + ch["start"], s + ch["end"]
                    text = src.content[start:end]
                    hay = text.lower()
                    scored.append({"source_id": sid, "start": start, "end": end, "text": text,
                                   "score": sum(w in hay for w in qwords) / max(len(qwords), 1)})
            scored.sort(key=lambda p: -p["score"])
            passages.extend(scored[: self.cfg["passages_per_source"]])
        passages.sort(key=lambda p: -p["score"])
        passages = passages[: self.cfg["max_passages"]]
        if not passages:
            raise AgentFailure(FailureType.NO_USEFUL_SOURCES, "no usable passages in the retrieved sources")
        return {"passages": passages, "flagged_sources": flagged, "fetch_errors": fetch_errors, "source_ids": ids}
