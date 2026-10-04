from datetime import datetime, timedelta, timezone

import pytest

from research_os.agents.supervisor import Supervisor
from research_os.contracts import RouterOutput, RoutingMode, Source, source_id_for_url
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.memory.embeddings import HashEmbedder
from research_os.memory.hooks import attach_memory, finalize_run
from research_os.memory.research_store import ResearchStore
from research_os.tools import build_registry
from research_os.tools.registry import ToolError


@pytest.fixture()
def store():
    s = ResearchStore("sqlite://", embedder=HashEmbedder())
    s.create_all()
    return s


class R:
    def __init__(self, workflow="general_research", intent="GENERAL_RESEARCH"):
        self.workflow, self.intent = workflow, intent

    def route(self, text):
        return RouterOutput(intent=self.intent, confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow=self.workflow, latency_ms=1,
                            entities={"technologies": [], "companies": [], "topics": [], "time_scope": []})


URL = "https://docs.x.dev/lg"
CONTENT = "LangGraph supports persistence through checkpointers. LangGraph also supports streaming."
SRC = Source(source_id=source_id_for_url(URL), title="LangGraph", url=URL, source_type="documentation",
             published_at=datetime.now(timezone.utc), retrieved_at=datetime.now(timezone.utc), content=CONTENT, tool="web_search")


def claim(cid, text, quote, verdict="SUPPORTED"):
    return {"claim_id": cid, "claim": text, "source_id": SRC.source_id, "evidence_quote": quote, "extractor_confidence": 0.9,
            "verdict": verdict, "verdict_rationale": "ok"}


class Pipeline:
    """Stub agents that add a source, then emit claims at the verify step."""

    def __init__(self, claims, crash_at=None):
        self.claims, self.crash_at = claims, crash_at

    def run(self, step, inputs, ctx, request):
        if step.agent == "search_agent":
            ctx.sources.add(SRC)
            return {"source_ids": [SRC.source_id]}
        if step.agent == "verifier":
            return {"claims": self.claims, "conflicts": [], "gate": {"result": "PASS", "reason": None, "failure_type": None}}
        if step.agent == "synthesis_agent":
            if self.crash_at == "synthesis":
                raise RuntimeError("synthesis crashed")
            return {"summary": "LangGraph persists state via checkpointers.", "findings": [], "claim_ids": []}
        return {"ok": step.id}


def agents(p):
    return {n: p for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}


def new_ctx(store, run_id, llm=None):
    c = RunContext.create(run_id, {}, llm=llm or ScriptedLLM([]), tools=build_registry(mode="fixture", memory=store), memory=store)
    attach_memory(c, store, "placeholder question", llm_backend="mock")
    return c


def run(store, run_id, text, p, router=None, llm=None):
    c = RunContext.create(run_id, {}, llm=llm or ScriptedLLM([]), tools=build_registry(mode="fixture", memory=store), memory=store)
    attach_memory(c, store, text, llm_backend="mock")
    out = build_graph(router or R(), Supervisor(), c, agents=agents(p)).invoke(initial_state(text))
    finalize_run(c, store, out)
    return out, c


def test_run_is_persisted_with_sources_claims_and_report(store):
    out, _ = run(store, "r1", "What does LangGraph persistence do?", Pipeline([claim("c1", "LangGraph supports persistence",
                                                                                    "LangGraph supports persistence through checkpointers")]))
    assert out["status"] == "SUCCESS"
    r = store.get_run("r1")
    assert r["status"] == "SUCCESS" and r["workflow"] == "general_research" and r["budget"]["steps"] == 5
    assert [c["claim_id"] for c in store.get_claims("r1")] == ["c1"] and store.get_sources("r1")[0].url == URL
    assert store.get_report("r1")["executive_summary"].startswith("LangGraph persists")


def test_partial_runs_are_persisted_and_a_crash_keeps_incrementally_saved_data(store):
    out, _ = run(store, "r2", "q about langgraph", Pipeline([claim("c1", "LangGraph supports persistence",
                                                                  "LangGraph supports persistence through checkpointers")], crash_at="synthesis"))
    assert out["status"] == "PARTIAL" and store.get_run("r2")["status"] == "PARTIAL"
    assert store.get_report("r2")["status"] == "PARTIAL" and len(store.get_claims("r2")) == 1
    # a hard crash before finalize: claims and sources were already saved by the hooks
    c = new_ctx(store, "r3")
    c.sources.add(SRC)
    c.step_listeners[0](type("S", (), {"agent": "verifier", "id": "verify"})(), {"claims": [claim("c7", "x persistence", "LangGraph supports persistence")]})
    assert store.get_run("r3")["status"] == "RUNNING" and len(store.get_sources("r3")) == 1 and store.get_claims("r3")[0]["claim_id"] == "c7"


def test_memory_search_tool_respects_permissions(store):
    run(store, "r1", "What does LangGraph persistence do?", Pipeline([]))
    c = new_ctx(store, "r9")
    res = c.tools.call("memory_search", {"query": "langgraph persistence", "kinds": ["questions"], "k": 3, "min_similarity": 0.3}, c, agent="supervisor")
    assert res.data["results"][0]["run_id"] == "r1"
    with pytest.raises(ToolError) as e:
        c.tools.call("memory_search", {"query": "langgraph"}, c, agent="search_agent")
    assert "permission" in str(e.value).lower()


def test_recall_request_is_answered_from_memory_without_llm_or_research_tools(store):
    run(store, "r1", "What does LangGraph persistence do?", Pipeline([claim("c1", "LangGraph supports persistence",
                                                                           "LangGraph supports persistence through checkpointers")]))
    llm = ScriptedLLM([])
    out, c = run(store, "r2", "What did we previously find about langgraph persistence?", Pipeline([]), llm=llm)
    assert out["status"] == "SUCCESS" and out.get("plan") is None
    assert out["recall"][0]["run_id"] == "r1" and out["recall"][0]["report"]["executive_summary"].startswith("LangGraph persists")
    assert llm.calls == 0 and c.budget.usage()["searches"] == 0 and c.budget.usage()["steps"] == 0
    assert store.get_run("r2")["status"] == "SUCCESS" and store.get_report("r2")["run_metadata"]["recalled_from"] == ["r1"]


def test_recall_with_nothing_in_memory_falls_through_to_normal_research(store):
    out, _ = run(store, "r1", "What did we previously find about quantum gravity?", Pipeline([]))
    assert out.get("recall") is None and out["status"] == "SUCCESS" and out["plan"] is not None


def test_non_recall_questions_are_not_hijacked_by_similar_past_questions(store):
    run(store, "r1", "What does LangGraph persistence do?", Pipeline([]))
    out, _ = run(store, "r2", "What does LangGraph persistence do?", Pipeline([]))      # same question, but not phrased as recall
    assert out.get("recall") is None and out["plan"] is not None


def test_prior_findings_are_offered_to_the_planner_when_fresh_and_valid(store):
    run(store, "r1", "What does LangGraph persistence do?", Pipeline([claim("c1", "LangGraph supports persistence",
                                                                           "LangGraph supports persistence through checkpointers")]))
    out, _ = run(store, "r2", "Explain langgraph persistence support", Pipeline([]))
    assert out["prior_findings"] and out["prior_findings"][0]["claim"]["claim_id"] == "c1"
    assert any("prior finding" in n for n in out["plan"].notes)


def test_stale_prior_findings_are_not_offered(store):
    store.create_run("old", "What does LangGraph persistence do?", "mock", created_at=datetime.now(timezone.utc) - timedelta(days=120))
    store.add_sources("old", [SRC])
    store.add_claims("old", [claim("c1", "LangGraph supports persistence", "LangGraph supports persistence through checkpointers")])
    store.finish_run("old", "SUCCESS", report={"summary": "old"})
    out, _ = run(store, "r2", "Explain langgraph persistence support", Pipeline([]))
    assert not out.get("prior_findings")


def test_monitoring_run_reports_new_vs_unchanged_findings(store):
    first = [claim("a", "LangGraph supports persistence", "LangGraph supports persistence through checkpointers")]
    run(store, "m1", "monitor langgraph releases", Pipeline(first), router=R("research_monitoring", "RESEARCH_MONITORING"))
    second = first + [claim("b", "LangGraph also supports streaming", "LangGraph also supports streaming")]
    out, _ = run(store, "m2", "monitor langgraph releases", Pipeline(second), router=R("research_monitoring", "RESEARCH_MONITORING"))
    d = out["monitoring_diff"]
    assert d["prior_run_id"] == "m1" and [c["claim_id"] for c in d["unchanged"]] == ["a"] and [c["claim_id"] for c in d["new"]] == ["b"]
