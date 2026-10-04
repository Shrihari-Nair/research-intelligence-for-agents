"""Search Agent: step objective + entities -> queries -> tools -> ranked, deduplicated Sources."""
from __future__ import annotations

from datetime import datetime, timezone

from research_os.agents.base import AgentFailure, agent_cfg
from research_os.agents.text import keywords
from research_os.contracts import FailureType, ResearchStep
from research_os.tools.registry import ToolError

SCHOLARLY = ("arxiv_search", "pubmed_search")


class SearchAgent:
    """Cheap heuristics only (no LLM): query building, overlap/recency/type ranking."""

    name = "search_agent"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or agent_cfg())["search"]

    @staticmethod
    def build_query(tool: str, entity: str | None, request: str) -> str:
        """Tool-appropriate query text."""
        words = keywords(request, 6)
        if entity:
            rest = [w for w in words if w not in entity.lower().split()]
            if tool == "github_search":
                return entity
            return " ".join([entity, *rest[:4]])
        if tool == "github_search":
            return " ".join(words[:4]) or request[:100]
        return request[:300] if tool == "web_search" else (" ".join(words) or request[:100])

    def _score(self, src, qwords: list[str], allowed: list[str]) -> float:
        hay = f"{src.title} {src.content}".lower()
        score = sum(w in hay for w in qwords) / max(len(qwords), 1)
        if allowed and src.source_type in allowed:
            score += 1.0  # allowed source types dominate topical overlap
        if src.published_at is not None:
            age_days = (datetime.now(timezone.utc) - src.published_at).days
            score += 0.5 if age_days < 730 else 0.0
        return score

    def run(self, step: ResearchStep, inputs: dict, ctx, request: str) -> dict:
        p = step.params
        entity, req, recent = p.get("entity"), p.get("request") or request, bool(p.get("recent"))
        found, queries, errors = [], [], []
        for tool in step.tools:
            level = int(p.get("broaden_level", 0))
            q = self.build_query(tool, entity, req)
            if level >= 2:
                q = " ".join([*keywords(req, 1), "overview"])
            elif level == 1:
                q = " ".join(keywords(req, 3)) or q
            args = {"query": q, "max_results": min(self.cfg["max_results_per_tool"] * (1 + level), 20)}
            if recent and tool in SCHOLARLY:
                args["sort"] = "recent"
            queries.append({"tool": tool, "query": q})
            try:
                found.extend(ctx.tools.call(tool, args, ctx, agent=self.name).sources)
            except ToolError as e:
                errors.append({"tool": tool, "failure_type": e.failure_type.value, "error": str(e)})
        if not found:
            if errors and len(errors) == len(step.tools):
                first = errors[0]
                raise ToolError(FailureType(first["failure_type"]), f"all search tools failed; first: {first['error']}")
            raise AgentFailure(FailureType.NO_USEFUL_SOURCES, f"no sources found for queries {[q['query'] for q in queries]}")
        qwords = keywords(" ".join([entity or "", req]), 8)
        skip = set(p.get("skip_source_ids", []))
        unique = {s.source_id: s for s in found if s.source_id not in skip}
        ranked = sorted(unique.values(), key=lambda s: -self._score(s, qwords, p.get("allowed_source_types", [])))
        top = ranked[: self.cfg["max_sources"]]
        return {"source_ids": [s.source_id for s in top], "queries": queries, "errors": errors}
