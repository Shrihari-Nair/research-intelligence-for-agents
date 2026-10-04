from research_os.agents import build_agents
from research_os.agents.supervisor import Supervisor
from research_os.contracts import Claim, RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.demo_client import DemoLLM
from research_os.tools import build_registry


class Router:
    def __init__(self, workflow, intent, techs=(), companies=()):
        self.args = (workflow, intent, list(techs), list(companies))

    def route(self, text):
        w, i, t, c = self.args
        return RouterOutput(intent=i, confidence=0.95, routing_mode=RoutingMode.DIRECT, workflow=w, latency_ms=1,
                            entities={"technologies": t, "companies": c, "topics": [], "time_scope": []})


def run(router, text):
    ctx = RunContext.create("demo", {}, llm=DemoLLM(), tools=build_registry(mode="fixture"))
    return build_graph(router, Supervisor(), ctx, agents=build_agents()).invoke(initial_state(text)), ctx


def test_demo_llm_drives_the_full_pipeline_offline_with_verifiable_claims():
    out, ctx = run(Router("technology_comparison", "TECHNOLOGY_COMPARISON", ["LangGraph", "CrewAI"]), "Compare LangGraph and CrewAI")
    assert out["status"] in ("SUCCESS", "PARTIAL")
    verify = next(s for s in out["plan"].steps if s.id == "verify")
    claims = [Claim(**c) for c in verify.outputs["claims"]]
    assert claims and all(c.evidence_quote in ctx.sources.get(c.source_id).content for c in claims)   # hard rule 8 holds in demo mode too


def test_demo_llm_general_research_produces_a_cited_report():
    out, ctx = run(Router("general_research", "GENERAL_RESEARCH"), "tell me about raft consensus leader election")
    assert out["status"] == "SUCCESS"
    synth = next(s for s in out["plan"].steps if s.id == "synthesize").outputs
    assert synth["findings"] and all(f["claim_ids"] for f in synth["findings"])


def test_demo_llm_never_follows_instructions_in_sources():
    out, ctx = run(Router("general_research", "GENERAL_RESEARCH", ["LangGraph"]), "LangGraph tips for beginners")
    text = " ".join(s.outputs.get("summary", "") for s in out["plan"].steps if s.id == "synthesize") + " ".join(
        c["claim"] for s in out["plan"].steps if s.id == "verify" for c in s.outputs.get("claims", []))
    assert "admin mode" not in text and "100 times faster" not in text


def test_demo_llm_is_deterministic_and_labelled():
    a, _ = run(Router("general_research", "GENERAL_RESEARCH"), "tell me about raft consensus leader election")
    b, _ = run(Router("general_research", "GENERAL_RESEARCH"), "tell me about raft consensus leader election")
    assert [s.outputs for s in a["plan"].steps if s.id == "synthesize"] == [s.outputs for s in b["plan"].steps if s.id == "synthesize"]
    assert DemoLLM.BACKEND == "demo"
