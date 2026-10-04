import pytest

from research_os.contracts import CostEstimate, ExecutionBudget, FailureType, ResearchPlan, ResearchStep, RunStatus
from research_os.graph.executor import classify_failure, execute_plan
from research_os.harness.budgets import BudgetExhausted, BudgetTracker
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry


def budget(**kw):
    base = dict(max_steps=5, max_tool_calls=5, max_searches=5, max_llm_calls=5, max_runtime_seconds=60)
    return ExecutionBudget(**{**base, **kw})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize("kind,limit_field", [("steps", "max_steps"), ("tool_calls", "max_tool_calls"),
                                              ("searches", "max_searches"), ("llm_calls", "max_llm_calls")])
def test_each_counter_limit_is_enforced_and_refusal_does_not_increment(kind, limit_field):
    t = BudgetTracker(budget(**{limit_field: 2}))
    t.count(kind)
    t.count(kind)
    with pytest.raises(BudgetExhausted) as e:
        t.count(kind)
    assert e.value.kind == kind and e.value.limit == 2 and e.value.failure_type == FailureType.BUDGET_EXHAUSTED
    assert t.usage()[kind] == 2


def test_runtime_limit_uses_the_clock_and_applies_to_every_kind():
    clock = Clock()
    t = BudgetTracker(budget(max_runtime_seconds=10), clock=clock)
    t.count("steps")
    clock.t = 10.5
    for kind in ("steps", "tool_calls", "llm_calls", "searches"):
        with pytest.raises(BudgetExhausted) as e:
            t.count(kind)
        assert e.value.kind == "runtime"


def test_unlimited_until_limits_are_set_and_set_limits_applies_retroactively():
    t = BudgetTracker()
    for _ in range(50):
        t.count("llm_calls")
    t.set_limits(budget(max_llm_calls=10))
    with pytest.raises(BudgetExhausted):
        t.count("llm_calls")
    assert t.limits().max_llm_calls == 10


def test_tool_registry_checks_the_budget_before_calling():
    c = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    c.budget.set_limits(budget(max_tool_calls=1, max_searches=9))
    c.tools.call("web_search", {"query": "langgraph"}, c)
    with pytest.raises(BudgetExhausted):
        c.tools.call("web_search", {"query": "crewai"}, c)
    assert c.budget.usage()["tool_calls"] == 1


def test_search_limit_does_not_block_non_search_tools():
    c = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    c.budget.set_limits(budget(max_searches=1))
    c.tools.call("web_search", {"query": "langgraph"}, c)
    with pytest.raises(BudgetExhausted):
        c.tools.call("web_search", {"query": "crewai"}, c)
    c.tools.call("fetch_url", {"url": "https://docs.example-db.dev/raft"}, c)


def test_llm_wrapper_checks_the_budget_and_emits_a_budget_event():
    c = RunContext.create("r", {}, llm=ScriptedLLM(["a", "b"]), tools=build_registry(mode="fixture"))
    c.budget.set_limits(budget(max_llm_calls=1))
    c.llm.complete("p1")
    with pytest.raises(BudgetExhausted):
        c.llm.complete("p2")
    ev = [e for e in c.events.events if e.type == "budget"]
    assert ev and ev[-1].payload["kind"] == "llm_calls" and ev[-1].payload["limit"] == 1
    assert c.llm.base.calls == 1


def mk(id, deps=()):
    return ResearchStep(id=id, objective="o", agent="a", tools=[], depends_on=list(deps))


class Ok:
    def run(self, step, inputs, ctx, request):
        return {"ok": step.id}


def plan3():
    return ResearchPlan(plan_id="p", workflow="w", steps=[mk("a"), mk("b", ["a"]), mk("c", ["b"])], estimated_cost=CostEstimate())


def test_step_budget_halts_the_run_as_partial_not_a_crash():
    c = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    c.budget.set_limits(budget(max_steps=2))
    res = execute_plan(plan3(), {"a": Ok()}, c, "r")
    st = {s.id: s.status for s in res.plan.steps}
    assert st == {"a": "succeeded", "b": "succeeded", "c": "skipped"} or st["c"] in ("failed", "skipped")
    assert res.status == RunStatus.PARTIAL and any(f["failure_type"] == "BUDGET_EXHAUSTED" for f in res.failures)


def test_runtime_limit_triggered_by_a_slow_step_halts_remaining_steps():
    clock = Clock()
    c = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    c.budget = BudgetTracker(budget(max_runtime_seconds=5), clock=clock)

    class Slow:
        def run(self, step, inputs, ctx, request):
            clock.t += 10   # the step "takes" 10 s
            return {"ok": 1}
    res = execute_plan(plan3(), {"a": Slow()}, c, "r")
    assert res.plan.steps[0].status == "succeeded" and [s.status for s in res.plan.steps[1:]] == ["failed", "skipped"] \
        or [s.status for s in res.plan.steps[1:]] == ["skipped", "skipped"]
    assert res.status == RunStatus.PARTIAL


def test_budget_exhaustion_is_classified():
    assert classify_failure(BudgetExhausted("llm_calls", 3, 3)) == FailureType.BUDGET_EXHAUSTED


def test_usage_includes_elapsed_and_limits_for_report_metadata():
    t = BudgetTracker(budget(max_steps=3))
    t.count("steps")
    u = t.usage()
    assert u["steps"] == 1 and "elapsed_seconds" in u and t.snapshot()["limits"]["max_steps"] == 3


def test_a_refused_search_call_does_not_leave_a_tool_call_increment_behind():
    t = BudgetTracker(budget(max_searches=1, max_tool_calls=9))
    t.count_all({"tool_calls": 1, "searches": 1})
    with pytest.raises(BudgetExhausted):
        t.count_all({"tool_calls": 1, "searches": 1})
    assert t.usage()["tool_calls"] == 1 and t.usage()["searches"] == 1


def test_graph_applies_the_workflow_budget_and_a_tiny_override_ends_partial():
    from research_os.agents.supervisor import Supervisor
    from research_os.contracts import RouterOutput, RoutingMode
    from research_os.graph.workflow import build_graph, initial_state

    class R:
        def route(self, text):
            return RouterOutput(intent="FACT_LOOKUP", confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow="fact_lookup",
                                latency_ms=1, entities={"technologies": [], "companies": [], "topics": [], "time_scope": []})
    agents = {n: Ok() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    c = RunContext.create("r", {"budget": {"max_steps": 2, "max_tool_calls": 5, "max_searches": 5, "max_llm_calls": 5,
                                           "max_runtime_seconds": 60}}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    out = build_graph(R(), Supervisor(), c, agents=agents).invoke(initial_state("what license"))
    assert out["status"] == "PARTIAL" and c.budget.usage()["steps"] == 2
    c2 = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    build_graph(R(), Supervisor(), c2, agents=agents).invoke(initial_state("what license"))
    assert c2.budget.limits().max_searches == 4   # the 'small' profile for fact_lookup


def test_elapsed_stops_counting_once_the_run_is_frozen():
    now = [100.0]
    b = BudgetTracker(clock=lambda: now[0])
    now[0] = 130.0
    b.freeze()
    now[0] = 9999.0                                  # hours later, someone reads the finished run's usage
    assert b.usage()["elapsed_seconds"] == 30.0
