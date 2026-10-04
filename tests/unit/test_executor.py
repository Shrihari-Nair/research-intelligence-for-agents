import pytest

from research_os.agents.supervisor import Supervisor
from research_os.contracts import CostEstimate, FailureType, ResearchPlan, ResearchStep, RouterOutput, RoutingMode, RunStatus
from research_os.graph.executor import execute_plan
from research_os.graph.templates import load_workflows
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.registry import ToolError


def mk(id, agent, deps=()):
    return ResearchStep(id=id, objective="obj " + id, agent=agent, tools=[], depends_on=list(deps))


def plan(*steps):
    return ResearchPlan(plan_id="p", workflow="general_research", steps=list(steps), estimated_cost=CostEstimate())


class Stub:
    def __init__(self, name, fail=None, log=None):
        self.name, self.fail, self.log = name, fail, log if log is not None else []

    def run(self, step, inputs, ctx, request):
        self.log.append((step.id, sorted(inputs)))
        if self.fail:
            raise self.fail
        return {"ok": step.id}


def agents(log, **fails):
    return {n: Stub(n, fails.get(n), log) for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}


def ctx():
    return RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))


def linear():
    return plan(mk("search", "search_agent"), mk("read", "source_agent", ["search"]), mk("extract", "evidence_agent", ["read"]),
                mk("verify", "verifier", ["extract"]), mk("synthesize", "synthesis_agent", ["verify"]))


def test_steps_run_in_dependency_order_and_receive_dependency_outputs():
    log, c = [], ctx()
    res = execute_plan(linear(), agents(log), c, "req")
    assert [s for s, _ in log] == ["search", "read", "extract", "verify", "synthesize"]
    assert dict(log)["read"] == ["search"] and dict(log)["synthesize"] == ["verify"]
    assert res.status == RunStatus.SUCCESS and all(s.status == "succeeded" and s.attempts == 1 for s in res.plan.steps)
    assert res.outputs["search"] == {"ok": "search"} and res.plan.steps[0].outputs == {"ok": "search"}
    assert c.budget.usage()["steps"] == 5


def test_out_of_order_plan_is_executed_topologically():
    log = []
    p = plan(mk("synthesize", "synthesis_agent", ["a"]), mk("a", "search_agent"))
    execute_plan(p, agents(log), ctx(), "r")
    assert [s for s, _ in log] == ["a", "synthesize"]


def test_failed_step_marks_dependents_skipped_and_never_raises():
    log, c = [], ctx()
    res = execute_plan(linear(), agents(log, source_agent=ToolError(FailureType.TOOL_FAILURE, "boom")), c, "r")
    st = {s.id: s.status for s in res.plan.steps}
    assert st == {"search": "succeeded", "read": "failed", "extract": "skipped", "verify": "skipped", "synthesize": "skipped"}
    assert res.status == RunStatus.PARTIAL and res.failures[0]["failure_type"] == "TOOL_FAILURE"
    assert [s for s, _ in log] == ["search", "read"]
    assert [e.type for e in c.events.events if e.node == "step:read"] == ["started", "failed"]


def test_unexpected_exception_is_classified_as_agent_failure():
    res = execute_plan(linear(), agents([], search_agent=ZeroDivisionError("x")), ctx(), "r")
    assert res.failures[0]["failure_type"] == "AGENT_FAILURE" and res.status == RunStatus.FAILED


def test_independent_branch_survives_a_failure():
    p = plan(mk("a", "search_agent"), mk("b", "search_agent"), mk("c", "source_agent", ["a"]),
             mk("s", "synthesis_agent", ["b", "c"]))
    log = []
    ag = agents(log)
    ag["search_agent"] = type("S", (), {"name": "search_agent", "run": lambda self, step, i, c, r: (_ for _ in ()).throw(
        ToolError(FailureType.SEARCH_TIMEOUT, "t")) if step.id == "a" else {"ok": 1}})()
    res = execute_plan(p, ag, ctx(), "r")
    st = {s.id: s.status for s in res.plan.steps}
    assert st["b"] == "succeeded" and st["a"] == "failed" and st["c"] == "skipped" and st["s"] == "skipped"
    assert res.status == RunStatus.PARTIAL


def test_missing_agent_for_a_step_is_a_clean_failure():
    ag = agents([])
    del ag["verifier"]
    res = execute_plan(linear(), ag, ctx(), "r")
    assert {s.id: s.status for s in res.plan.steps}["verify"] == "failed"


@pytest.mark.parametrize("name", list(load_workflows()))
def test_every_workflow_plan_executes_end_to_end_through_the_graph(name):
    intent = load_workflows()[name]["intent"]

    class R:
        def route(self, text):
            return RouterOutput(intent=intent, confidence=0.95, routing_mode=RoutingMode.DIRECT, workflow=name, latency_ms=1,
                                entities={"technologies": ["LangGraph", "CrewAI"], "companies": ["OpenAI", "Anthropic"],
                                          "topics": [], "time_scope": []})
    c, log = ctx(), []
    g = build_graph(R(), Supervisor(), c, agents=agents(log))
    out = g.invoke(initial_state("research this"))
    assert out["status"] == "SUCCESS" and out["plan"].workflow == name
    assert [s for s, _ in log][-1].startswith("synthesize") and len(log) == len(out["plan"].steps)
