import pytest

from research_os.contracts import CostEstimate, FailureType, ResearchPlan, ResearchStep, RunStatus
from research_os.graph.executor import execute_plan
from research_os.harness.budgets import BudgetExhausted
from research_os.harness.context import RunContext
from research_os.harness.recovery import RecoveryManager, build_partial_artifact
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.registry import ToolError

FT = FailureType


def step(id, agent="search_agent", deps=(), tools=("web_search",)):
    return ResearchStep(id=id, objective="o", agent=agent, tools=list(tools), depends_on=list(deps), params={"request": "q"})


def plan(*steps):
    return ResearchPlan(plan_id="p", workflow="w", steps=list(steps), estimated_cost=CostEstimate())


def ctx():
    return RunContext.create("r", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))


class FailThenOk:
    """Raises `exc` for the first `n` calls (n=None: always), then succeeds. Records the step state at each call."""

    def __init__(self, exc, n=None):
        self.exc, self.n, self.calls, self.seen = exc, n, 0, []

    def run(self, step, inputs, ctx, request):
        self.calls += 1
        self.seen.append((list(step.tools), dict(step.params)))
        if self.n is None or self.calls <= self.n:
            raise self.exc
        return {"ok": step.id}


class Ok:
    def run(self, step, inputs, ctx, request):
        return {"ok": step.id}


def run(p, agents, rm=None, c=None):
    c = c or ctx()
    return execute_plan(p, agents, c, "q", recovery=rm or RecoveryManager()), c


def actions(c):
    return [e.payload["action"] for e in c.events.events if e.type == "recovery"]


def two_step(first_agent):
    return plan(step("a"), step("s", "synthesis_agent", ["a"], tools=())), {"search_agent": first_agent, "synthesis_agent": Ok()}


@pytest.mark.parametrize("ft,exc,expected,final", [
    (FT.SEARCH_TIMEOUT, ToolError(FT.SEARCH_TIMEOUT, "slow"), ["retry", "alternate_tool", "terminate_step"], "failed"),
    (FT.INVALID_SOURCE, ToolError(FT.INVALID_SOURCE, "404"), ["skip_source", "alternate_tool", "terminate_step"], "failed"),
    (FT.MALFORMED_LLM_OUTPUT, ToolError(FT.MALFORMED_LLM_OUTPUT, "junk"), ["retry_with_error", "terminate_step"], "failed"),
    (FT.TOOL_FAILURE, ToolError(FT.TOOL_FAILURE, "boom"), ["retry", "alternate_tool", "terminate_step"], "failed"),
    (FT.AGENT_FAILURE, RuntimeError("bug"), ["retry", "skip_step"], "skipped"),
])
def test_persistent_failure_follows_the_configured_action_sequence(ft, exc, expected, final):
    agent = FailThenOk(exc)
    (res, c) = run(*two_step(agent))
    assert actions(c) == expected
    assert {s.id: s.status for s in res.plan.steps}["a"] == final
    assert res.plan.steps[1].status == "skipped" and res.status == RunStatus.FAILED
    assert agent.calls == len(expected)                      # one initial run + one re-run per non-terminal action
    assert all(e.payload["failure_type"] == ft.value for e in c.events.events if e.type == "recovery")


def test_transient_failure_recovers_with_a_single_retry_and_the_run_succeeds():
    res, c = run(*two_step(FailThenOk(ToolError(FT.SEARCH_TIMEOUT, "slow"), n=1)))
    assert actions(c) == ["retry"] and res.status == RunStatus.SUCCESS
    assert res.recoveries[0]["outcome"] and res.plan.steps[0].attempts == 2


def test_alternate_tool_swaps_the_failing_tool_for_the_configured_alternate():
    agent = FailThenOk(ToolError(FT.SEARCH_TIMEOUT, "slow"), n=2)
    res, c = run(*two_step(agent))
    assert actions(c) == ["retry", "alternate_tool"] and res.status == RunStatus.SUCCESS
    assert agent.seen[0][0] == ["web_search"] and agent.seen[2][0] == ["github_search"]


def test_malformed_output_retry_passes_the_validation_error_as_a_hint():
    agent = FailThenOk(ToolError(FT.MALFORMED_LLM_OUTPUT, "missing field 'claims'"), n=1)
    res, _ = run(*two_step(agent))
    assert "missing field" in agent.seen[1][1]["last_error"] and res.status == RunStatus.SUCCESS


def test_no_useful_sources_broadens_then_alternates_then_requests_a_replan():
    from research_os.contracts import FailureType as F
    agent = FailThenOk(ToolError(F.NO_USEFUL_SOURCES, "nothing"))
    res, c = run(*two_step(agent))
    assert actions(c) == ["broaden_query", "alternate_tool", "replan"]
    assert agent.seen[1][1]["broaden_level"] == 1 and res.replan_requested is True and res.status == RunStatus.FAILED


def test_after_the_replan_is_used_up_the_next_failure_asks_the_user():
    rm = RecoveryManager()
    agent = FailThenOk(ToolError(FT.NO_USEFUL_SOURCES, "nothing"))
    res, c = run(*two_step(agent), rm=rm)
    assert res.replan_requested
    res2, c2 = run(*two_step(agent), rm=rm)                  # the graph re-runs the plan with the same manager
    assert actions(c2) == ["ask_user"] and res2.ask_user and res2.status in (RunStatus.PARTIAL, RunStatus.FAILED)
    assert "rephrase" in res2.ask_user.lower() or "narrow" in res2.ask_user.lower()


def test_budget_exhaustion_terminates_safely_and_halts_everything_after():
    p = plan(step("a"), step("b", deps=["a"]), step("s", "synthesis_agent", ["b"], tools=()))
    res, c = run(p, {"search_agent": FailThenOk(BudgetExhausted("llm_calls", 3, 3), n=1), "synthesis_agent": Ok()})
    assert actions(c) == ["terminate_safely"] and res.halted_reason and res.status == RunStatus.PARTIAL
    assert [s.status for s in res.plan.steps] == ["failed", "skipped", "skipped"]


def test_conflicting_evidence_proceeds_and_is_not_a_failure():
    rm = RecoveryManager()
    d = rm.decide(step("verify", "verifier"), FT.CONFLICTING_EVIDENCE, ctx())
    assert d.action == "proceed"


def test_impossible_plan_replans_once_then_terminates_safely():
    rm, c = RecoveryManager(), ctx()
    s = step("plan")
    assert rm.decide(s, FT.IMPOSSIBLE_PLAN, c).action == "replan"
    assert rm.decide(s, FT.IMPOSSIBLE_PLAN, c).action == "terminate_safely"


# ---------- caps: recovery can never loop forever ----------
def test_per_step_cap_stops_a_retry_loop():
    rm = RecoveryManager(policy={"limits": {"max_per_step": 3, "max_per_run": 99, "max_replans": 1}, "alternates": {},
                                 "policy": {"TOOL_FAILURE": ["retry"] * 50}})
    agent = FailThenOk(ToolError(FT.TOOL_FAILURE, "boom"))
    res, c = run(*two_step(agent), rm=rm)
    assert agent.calls == 4 and res.plan.steps[0].status == "failed"        # initial + 3 recoveries, then terminated
    assert actions(c).count("retry") == 3 and actions(c)[-1] == "terminate_step"


def test_per_run_cap_halts_the_whole_run_safely():
    rm = RecoveryManager(policy={"limits": {"max_per_step": 99, "max_per_run": 3, "max_replans": 1}, "alternates": {},
                                 "policy": {"TOOL_FAILURE": ["retry"] * 50}})
    agent = FailThenOk(ToolError(FT.TOOL_FAILURE, "boom"))
    res, c = run(*two_step(agent), rm=rm)
    assert agent.calls == 4 and res.halted_reason and "cap" in res.halted_reason and res.plan.steps[1].status == "skipped"
    assert actions(c)[-1] == "terminate_safely"


def test_recovery_reruns_consume_budget():
    from research_os.contracts import ExecutionBudget
    c = ctx()
    c.budget.set_limits(ExecutionBudget(max_steps=3, max_tool_calls=9, max_searches=9, max_llm_calls=9, max_runtime_seconds=60))
    agent = FailThenOk(ToolError(FT.TOOL_FAILURE, "boom"))
    many = RecoveryManager(policy={"limits": {"max_per_step": 50, "max_per_run": 50, "max_replans": 1}, "alternates": {},
                                   "policy": {"TOOL_FAILURE": ["retry"] * 50}})
    res, _ = run(*two_step(agent), rm=many, c=c)
    assert c.budget.usage()["steps"] == 3                    # the 4th attempt was refused by the step budget
    assert res.halted_reason and "budget" in res.halted_reason.lower()


def test_unknown_failure_type_policy_falls_back_to_terminate_step():
    rm = RecoveryManager(policy={"limits": {"max_per_step": 3, "max_per_run": 9, "max_replans": 1}, "alternates": {}, "policy": {}})
    assert rm.decide(step("a"), FT.TOOL_FAILURE, ctx()).action == "terminate_step"


# ---------- partial artifact ----------
def test_terminate_safely_always_yields_a_valid_partial_artifact():
    p = plan(step("a"), step("s", "synthesis_agent", ["a"], tools=()))
    res, c = run(p, {"search_agent": FailThenOk(BudgetExhausted("steps", 1, 1)), "synthesis_agent": Ok()})
    art = build_partial_artifact(res, c, "my question")
    assert art["status"] == "PARTIAL" and art["question"] == "my question"
    assert art["completed_steps"] == [] and {x["id"] for x in art["incomplete_steps"]} == {"a", "s"}
    assert art["message"] and art["claims"] == [] and "budget" in art["halted_reason"].lower()


def test_budget_exhaustion_always_terminates_safely_even_if_the_policy_file_omits_it():
    rm = RecoveryManager(policy={"limits": {"max_per_step": 9, "max_per_run": 9, "max_replans": 1}, "alternates": {}, "policy": {}})
    assert rm.decide(step("a"), FT.BUDGET_EXHAUSTED, ctx()).action == "terminate_safely"


def test_a_replanned_execution_keeps_the_first_executions_failures_and_recoveries():
    from research_os.graph.executor import merge_history
    agent = FailThenOk(ToolError(FT.NO_USEFUL_SOURCES, "nothing"))
    first, _ = run(*two_step(agent))
    second, _ = run(*two_step(agent), rm=RecoveryManager())
    n_first, n_second = len(first.failures), len(second.failures)
    merged = merge_history(first, second)
    assert len(merged.failures) == n_first + n_second
    assert [r["action"] for r in merged.recoveries][: len(first.recoveries)] == [r["action"] for r in first.recoveries]
    assert "replan" in [r["action"] for r in merged.recoveries]
