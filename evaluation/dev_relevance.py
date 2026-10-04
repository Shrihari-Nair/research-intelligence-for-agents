"""Dev harness for passage relevance (NOT the 120-task evaluation set; these 20 questions are only for tuning the source agent).

`collect` (live, needs TAVILY_API_KEY) caches the sources the real search+fetch path returns for each question into
evaluation/dev_relevance_sources.json. `evaluate` then re-runs only the SourceAgent passage selection on that cache (offline, no LLM)
and reports whether a passage containing the known answer term reaches the evidence step.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from research_os.agents.search_agent import SearchAgent
from research_os.agents.source_agent import SourceAgent
from research_os.common import load_dotenv
from research_os.contracts import ResearchStep, Source
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry

DEV = Path("evaluation/dev_relevance.jsonl")
CACHE = Path("evaluation/dev_relevance_sources.json")


def topic_pattern(q: str) -> str:
    """Crude on-topic proxy for dev tuning only: words that signal a passage speaks to the kind of fact asked for."""
    ql = q.lower()
    for key, pat in (("licen", r"licen[sc]"), ("port", r"\bport\b|\b\d{4,5}\b"), ("released", r"releas|launch|debut"),
                     ("created", r"creat|invent|develop|author|found"), ("company", r"develop|creat|maintain|open[- ]source"),
                     ("written in", r"written|language|implement"), ("database", r"database|clickhouse|postgres|store")):
        if key in ql:
            return pat
    return r"."


def load_dev() -> list[dict]:
    return [json.loads(line) for line in DEV.read_text().splitlines() if line.strip()]


def _step(agent: str, tools: list[str], q: str, deps: list[str] | None = None) -> ResearchStep:
    return ResearchStep(id=agent, objective=f"Find and read sources answering: {q}", agent=agent, tools=tools, depends_on=deps or [],
                        params={"request": q})


def collect() -> None:
    load_dotenv()
    out = {}
    for item in load_dev():
        q = item["question"]
        ctx = RunContext.create("dev", {}, llm=ScriptedLLM([]), tools=build_registry(mode="live"))
        found = SearchAgent().run(_step("search", ["web_search"], q), {}, ctx, q)
        SourceAgent().run(_step("read", ["fetch_url"], q, ["search"]), {"search": found}, ctx, q)     # fetches full pages into the store
        out[q] = [ctx.sources.get(i).model_dump(mode="json") for i in found["source_ids"] if ctx.sources.get(i)]
    CACHE.write_text(json.dumps(out, indent=1))


def evaluate(agent: SourceAgent | None = None, k: int | None = None) -> dict:
    agent = agent or SourceAgent()
    cache = json.loads(CACHE.read_text())
    rows = []
    for item in load_dev():
        q, terms = item["question"], item["answer_terms"]
        ctx = RunContext.create("dev", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
        for s in cache[q]:
            ctx.sources.add(Source.model_validate(s))
        try:
            ps = agent.run(_step("read", [], q, ["search"]), {"search": {"source_ids": [s["source_id"] for s in cache[q]]}}, ctx, q)["passages"]
        except Exception:  # noqa: BLE001  (no usable passages -> counts as a miss)
            ps = []
        ps = ps[:k] if k else ps
        pats = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms]
        rank = next((i for i, p in enumerate(ps) if any(pt.search(p["text"]) for pt in pats)), None)
        tp = re.compile(topic_pattern(q), re.I)
        on_topic = [bool(tp.search(p["text"])) or any(pt.search(p["text"]) for pt in pats) for p in ps]
        rows.append({"question": q, "n_passages": len(ps), "first_answer_rank": rank, "answer_in_passages": rank is not None,
                     "precision": (sum(on_topic) / len(on_topic)) if on_topic else None,
                     "answer_in_any_source": any(any(pt.search(s["content"]) for pt in pats) for s in cache[q])})
    n = len(rows)
    reachable = [r for r in rows if r["answer_in_any_source"]]
    prec = [r["precision"] for r in rows if r["precision"] is not None]
    return {"n": n, "mean_precision": round(sum(prec) / len(prec), 3) if prec else None, "no_passages": sum(r["n_passages"] == 0 for r in rows),
            "answer_in_any_source": len(reachable), "answer_reaches_evidence": sum(r["answer_in_passages"] for r in reachable),
            "mean_passages": round(sum(r["n_passages"] for r in rows) / n, 1), "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "evaluate"])
    a = ap.parse_args()
    if a.cmd == "collect":
        collect()
    else:
        r = evaluate()
        print({k: v for k, v in r.items() if k != "rows"})
        for row in r["rows"]:
            print(row["answer_in_passages"], row["answer_in_any_source"], row["first_answer_rank"], row["n_passages"], row["question"])
