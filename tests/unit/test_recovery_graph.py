from research_os.agents.planner import Planner, PlanValidationError
from research_os.agents.supervisor import Supervisor
from research_os.contracts import FailureType as FT
from research_os.contracts import RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.harness.recovery import RecoveryManager
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.registry import ToolError


class R:
    def __init__(self, workflow="fact_lookup", intent="FACT_LOOKUP"):
        self.workflow, self.intent = workflow, intent

    def route(self, text):
        return RouterOutput(intent=self.intent, confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow=self.workflow, latency_ms=1,
                            entities={"technologies": ["LangGraph"], "companies": [], "topics": [], "time_scope": []})


class Ok:
    def run(self, step, inputs, ctx, request):
        return {"ok": step.id, "claims": []}


class Always:
    def __init__(self, exc):
        self.exc, self.calls, self.params = exc, 0, []

    def run(self, step, inputs, ctx, request):
        self.calls += 1
        self.params.append(dict(step.params))
        raise self.exc


class GateThenPass:
    """First verification fails its gate; after the replan it passes."""

    def __init__(self):
        self.calls = 0

    def run(self, step, inputs, ctx, request):
        self.calls += 1
        if self.calls == 1:
            return {"claims": [], "gate": {"result": "FAIL", "reason": "supported claims (0) below min_supported_claims (1)",
                                           "failure_type": "NO_USEFUL_SOURCES"}}
        return {"claims": [], "gate": {"result": "PASS", "reason": None, "failure_type": None}}


def agents(**over):
    base = {n: Ok() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    return {**base, **over}


def ctx():
    return RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))


def actions(c):
    return [e.payload["action"] for e in c.events.events if e.type == "recovery"]


def go(ag, c=None, router=None, planner=None):
    c = c or ctx()
    g = build_graph(router or R(), Supervisor(), c, agents=ag, planner=planner, recovery=RecoveryManager())
    return g.invoke(initial_state("what license does langgraph use")), c


def test_no_useful_sources_walks_broaden_alternate_replan_then_asks_the_user():
    search = Always(ToolError(FT.NO_USEFUL_SOURCES, "nothing"))
    out, c = go(agents(search_agent=search))
    assert actions(c) == ["broaden_query", "alternate_tool", "replan", "ask_user"]
    assert out["status"] == "PARTIAL" and out["partial_artifact"]["ask_user"] and out["partial_artifact"]["status"] == "PARTIAL"
    assert search.params[-1].get("broaden_level", 0) >= 1                   # the replanned search was broader
    assert search.calls == 4                                              # 3 runs before the replan + 1 after it, then stop


def test_verifier_gate_fail_triggers_a_replan_and_the_second_pass_succeeds():
    ver = GateThenPass()
    out, c = go(agents(verifier=ver))
    assert actions(c) == ["replan"] and out["status"] == "SUCCESS" and ver.calls == 2
    assert "partial_artifact" not in out or out.get("partial_artifact") is None


class BrokenPlanner(Planner):
    def __init__(self, fail_times):
        super().__init__()
        self.fail_times, self.calls = fail_times, 0

    def plan(self, request, decision, routed, ctx):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise PlanValidationError("synthetic invalid plan")
        return super().plan(request, decision, routed, ctx)


def test_impossible_plan_replans_once_with_the_general_research_fallback_and_runs():
    p = BrokenPlanner(1)
    out, c = go(agents(), router=R("technology_comparison", "TECHNOLOGY_COMPARISON"), planner=p)
    assert actions(c) == ["replan"] and out["status"] == "SUCCESS" and out["plan"].workflow == "general_research"
    assert any("fallback" in n or "replanned" in n for n in out["plan"].notes)


def test_impossible_plan_twice_terminates_safely_with_a_valid_partial_artifact():
    out, c = go(agents(), planner=BrokenPlanner(5))
    assert actions(c) == ["replan", "terminate_safely"]
    art = out["partial_artifact"]
    assert out["status"] == "PARTIAL" and art["status"] == "PARTIAL" and art["completed_steps"] == [] and "plan" in art["message"].lower()


def test_budget_exhaustion_in_the_graph_yields_a_partial_artifact_with_completed_work():
    class Eat:
        def run(self, step, inputs, ctx, request):
            ctx.budget.count("llm_calls", 99)
            return {}
    c = ctx()
    out, _ = go(agents(search_agent=Ok(), source_agent=Ok(), evidence_agent=Eat()), c=c)
    assert out["status"] == "PARTIAL" and actions(c) == ["terminate_safely"]
    assert out["partial_artifact"]["completed_steps"] == ["search", "read"] and "budget" in out["partial_artifact"]["halted_reason"].lower()
