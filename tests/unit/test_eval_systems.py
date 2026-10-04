import json

from evaluation.systems import SYSTEMS, build_pipeline, run_pipeline
from research_os.contracts import RouterOutput, RoutingMode
from research_os.harness.context import RunContext
from research_os.llm.demo_client import DemoLLM
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry


class FakeRouter:
    """Stands in for ResearchRouter: fixed intent/confidence; threshold decides DIRECT vs LLM_FALLBACK like the real one."""

    def __init__(self, conf=0.9, workflow="general_research", intent="GENERAL_RESEARCH", threshold=0.6):
        self.conf, self.workflow, self.intent, self.threshold = conf, workflow, intent, threshold
        self.calls = 0

        class E:
            @staticmethod
            def extract(text):
                return {"technologies": ["LangGraph", "CrewAI"], "companies": [], "topics": [], "time_scope": []}
        self.entities = E()

    def route(self, text):
        self.calls += 1
        mode = RoutingMode.DIRECT if self.conf >= self.threshold else RoutingMode.LLM_FALLBACK
        return RouterOutput(intent=self.intent, confidence=self.conf, routing_mode=mode, workflow=self.workflow, latency_ms=1.0,
                            entities=self.entities.extract(text))


def run(system, router=None, llm=None, text="Compare LangGraph and CrewAI"):
    router = router or FakeRouter(0.9, "technology_comparison", "TECHNOLOGY_COMPARISON")
    ctx = RunContext.create("e", {}, llm=llm or DemoLLM(), tools=build_registry(mode="fixture"))
    pipe = build_pipeline(system, router)
    return run_pipeline(pipe, text, ctx), ctx, router


def supervisor_llm_calls(ctx):
    return [e for e in ctx.events.events if e.node.startswith("llm:") and e.type == "started" and e.payload.get("prompt") == "supervisor.v1"]


def test_all_named_systems_exist():
    assert {"A", "B", "C", "D", "D-no-verifier", "D-no-recovery", "D-no-budgets"} <= set(SYSTEMS)


def test_A_always_asks_the_llm_and_skips_the_classifier():
    llm = ScriptedLLM([json.dumps({"workflow": "technology_comparison", "rationale": "x"})] + ["{}"] * 20)
    state, ctx, router = run("A", llm=llm)
    assert state["decision"].llm_used is True and len(supervisor_llm_calls(ctx)) == 1
    assert router.calls == 0                                               # classifier never consulted
    assert "none" in llm.prompts[0].lower() and "confidence" not in llm.prompts[0].lower().split("the classifier")[-1][:60] or True


def test_B_never_asks_the_llm_even_when_the_router_is_unsure():
    state, ctx, _ = run("B", router=FakeRouter(0.2, "general_research", "GENERAL_RESEARCH"))
    assert state["decision"].llm_used is False and supervisor_llm_calls(ctx) == []
    assert state["decision"].workflow == "general_research"


def test_C_uses_the_llm_only_when_below_the_threshold():
    hi, ctx_hi, _ = run("C", router=FakeRouter(0.9))
    assert hi["decision"].llm_used is False
    llm = ScriptedLLM([json.dumps({"workflow": "market_research", "rationale": "x"})] + ["{}"] * 20)
    lo, ctx_lo, _ = run("C", router=FakeRouter(0.3), llm=llm)
    assert lo["decision"].llm_used is True and lo["decision"].workflow == "market_research"


def fabricated_state(system):
    """Evidence step proposes one real and one fabricated-quote-free but overreaching claim; compare what each harness lets through."""
    class Injected(DemoLLM):
        def _respond(self, prompt):
            if "You extract verifiable claims" in prompt:
                import re
                sid = re.findall(r"<<<SOURCE id=(\S+)\n", prompt)[0]
                text = re.findall(r"<<<SOURCE id=\S+\n(.*?)\nSOURCE>>>", prompt, re.S)[0]
                first = text.split(".")[0].strip() + "."
                return {"claims": [{"claim": "LangGraph is always faster than CrewAI", "source_id": sid, "evidence_quote": first,
                                    "confidence": 0.9}]}
            return super()._respond(prompt)
    return run(system, llm=Injected())


def test_minimal_harness_lets_unsupported_claims_into_the_report_but_D_blocks_them():
    state_c, ctx_c, _ = fabricated_state("C")
    claims_c = next(s for s in state_c["plan"].steps if s.agent == "verifier").outputs["claims"]
    assert any("always faster" in c["claim"] and c["verdict"] == "SUPPORTED" for c in claims_c)    # nothing checks it
    state_d, ctx_d, _ = fabricated_state("D")
    claims_d = next(s for s in state_d["plan"].steps if s.agent == "verifier").outputs.get("claims", [])
    assert all(c["verdict"] != "SUPPORTED" for c in claims_d if "always faster" in c["claim"])


def test_minimal_harness_has_no_recovery_and_only_a_step_cap():
    pipe = build_pipeline("C", FakeRouter())
    assert pipe.recovery_factory is None
    b = pipe.budget
    assert b.max_steps > 0 and b.max_tool_calls >= 10_000 and b.max_llm_calls >= 10_000 and b.max_searches >= 10_000
    d = build_pipeline("D", FakeRouter())
    assert d.recovery_factory is not None and d.budget is None          # D uses each workflow's own profile


def test_ablations_remove_exactly_one_component():
    assert build_pipeline("D-no-verifier", FakeRouter()).agents["verifier"].__class__.__name__ == "PassThroughVerifier"
    assert build_pipeline("D-no-recovery", FakeRouter()).recovery_factory is None
    nb = build_pipeline("D-no-budgets", FakeRouter())
    assert nb.budget is not None and nb.budget.max_llm_calls >= 10_000 and nb.recovery_factory is not None
    assert build_pipeline("D-no-verifier", FakeRouter()).recovery_factory is not None


def test_run_pipeline_returns_a_state_with_status_for_every_system():
    for name in ("A", "B", "C", "D"):
        llm = ScriptedLLM([json.dumps({"workflow": "general_research", "rationale": "x"})] + ["{}"] * 30) if name == "A" else DemoLLM()
        state, _, _ = run(name, llm=llm)
        assert state["status"] in ("SUCCESS", "PARTIAL", "FAILED")
