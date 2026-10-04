import json

import pytest

from research_os.agents.planner import Planner, PlanValidationError, validate_plan
from research_os.agents.supervisor import SupervisorDecision
from research_os.contracts import CostEstimate, ExecutionBudget, FailureType, ResearchPlan, ResearchStep, RouterOutput, RoutingMode
from research_os.graph.templates import load_workflows
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry

TOOLS = build_registry(mode="fixture").names()
BIG = ExecutionBudget(max_steps=100, max_tool_calls=100, max_searches=100, max_llm_calls=100, max_runtime_seconds=100)


def routed(techs=(), companies=(), topics=()):
    return RouterOutput(intent="X", confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow="w", latency_ms=1,
                        entities={"technologies": list(techs), "companies": list(companies), "topics": list(topics),
                                  "time_scope": []})


def decision(workflow):
    return SupervisorDecision(workflow=workflow, intent=load_workflows()[workflow]["intent"], mode=RoutingMode.DIRECT,
                              rationale="t", llm_used=False)


def ctx(responses=()):
    return RunContext.create("r", {}, llm=ScriptedLLM(list(responses)), tools=build_registry(mode="fixture"))


def step(id, agent="search_agent", deps=(), tools=()):
    return ResearchStep(id=id, objective="o", agent=agent, tools=list(tools), depends_on=list(deps))


def plan_of(*steps):
    return ResearchPlan(plan_id="p", workflow="general_research", steps=list(steps), estimated_cost=CostEstimate())


def test_comparison_expands_one_step_per_technology_with_entity_aware_dependencies():
    p = Planner().plan("Compare LangGraph and CrewAI", decision("technology_comparison"), routed(["LangGraph", "CrewAI"]), ctx())
    ids = [s.id for s in p.steps]
    assert ids == ["search:langgraph", "search:crewai", "read:langgraph", "read:crewai", "extract:langgraph",
                   "extract:crewai", "verify", "synthesize"] or set(ids) == {
        "search:langgraph", "search:crewai", "read:langgraph", "read:crewai", "extract:langgraph", "extract:crewai",
        "verify", "synthesize"}
    by = {s.id: s for s in p.steps}
    assert by["read:langgraph"].depends_on == ["search:langgraph"]          # same-entity dependency
    assert by["verify"].depends_on == ["extract:langgraph", "extract:crewai"]  # single step depends on all instances
    assert "LangGraph" in by["search:langgraph"].objective and "Compare LangGraph" in by["search:langgraph"].objective
    assert p.workflow == "technology_comparison"


def test_no_entities_falls_back_to_a_single_instance_on_the_request():
    p = Planner().plan("compare these two things", decision("technology_comparison"), routed(), ctx())
    assert [s.id for s in p.steps] == ["search", "read", "extract", "verify", "synthesize"]
    assert "compare these two things" in p.steps[0].objective


def test_linear_workflow_for_fact_lookup():
    p = Planner().plan("what license does fastapi use", decision("fact_lookup"), routed(["FastAPI"]), ctx())
    assert [s.id for s in p.steps] == ["search", "read", "extract", "verify", "synthesize"]
    assert p.steps[-1].agent == "synthesis_agent" and all(s.status == "pending" for s in p.steps)


def test_cost_estimate_is_derived_from_the_plan():
    p = Planner().plan("what license does fastapi use", decision("fact_lookup"), routed(), ctx())
    c = p.estimated_cost
    assert c.searches == 2          # web_search + github_search in the one search step
    assert c.sources >= c.searches and c.llm_calls >= 2 and c.tool_calls >= c.searches
    bigger = Planner().plan("c", decision("technology_comparison"), routed(["A1", "B1", "C1"]), ctx()).estimated_cost
    assert bigger.searches == 6 and bigger.llm_calls > c.llm_calls


def test_many_entities_are_truncated_to_fit_the_budget_and_noted():
    techs = [f"Tech{i}A" for i in range(12)]
    p = Planner().plan("compare lots", decision("technology_comparison"), routed(techs), ctx())
    from research_os.graph.templates import budget_for
    assert len(p.steps) <= budget_for("technology_comparison").max_steps
    assert any("truncated" in n for n in p.notes)
    validate_plan(p, budget_for("technology_comparison"), TOOLS)  # still valid


@pytest.mark.parametrize("name", list(load_workflows()))
def test_every_workflow_produces_a_valid_plan(name):
    from research_os.graph.templates import budget_for
    p = Planner().plan("research this", decision(name), routed(["LangGraph", "CrewAI"], ["OpenAI", "Anthropic"]), ctx())
    validate_plan(p, budget_for(name), TOOLS)
    assert p.steps[-1].agent == "synthesis_agent"


# ---------- validation ----------
def expect_problem(p, text, budget=BIG):
    with pytest.raises(PlanValidationError) as e:
        validate_plan(p, budget, TOOLS)
    assert text in str(e.value) and e.value.failure_type == FailureType.IMPOSSIBLE_PLAN


def test_cycle_detected():
    expect_problem(plan_of(step("a", deps=["b"]), step("b", deps=["a"]), step("s", "synthesis_agent", ["a", "b"])), "cycle")


def test_missing_dependency_detected():
    expect_problem(plan_of(step("a", deps=["ghost"]), step("s", "synthesis_agent", ["a"])), "unknown dependency")


def test_duplicate_ids_detected():
    expect_problem(plan_of(step("a"), step("a"), step("s", "synthesis_agent", ["a"])), "duplicate")


def test_unknown_agent_and_tool_detected():
    expect_problem(plan_of(step("a", "wizard_agent"), step("s", "synthesis_agent", ["a"])), "unknown agent")
    expect_problem(plan_of(step("a", tools=["teleport"]), step("s", "synthesis_agent", ["a"])), "unknown tool")


def test_over_budget_detected():
    small = ExecutionBudget(max_steps=2, max_tool_calls=9, max_searches=9, max_llm_calls=9, max_runtime_seconds=9)
    expect_problem(plan_of(step("a"), step("b", deps=["a"]), step("s", "synthesis_agent", ["b"])), "max_steps", small)


def test_final_step_must_be_the_only_synthesis_and_reach_every_step():
    expect_problem(plan_of(step("s", "synthesis_agent"), step("a", deps=["s"])), "final step")
    expect_problem(plan_of(step("a"), step("orphan"), step("s", "synthesis_agent", ["a"])), "orphan")
    expect_problem(plan_of(), "no steps")


# ---------- optional bounded LLM refinement ----------
def refine_json(extra):
    return json.dumps({"extra_steps": extra})


def extra_step(id="extra1", tools=("web_search",), agent="search_agent", deps=()):
    return {"id": id, "objective": "also look at X", "agent": agent, "tools": list(tools), "depends_on": list(deps)}


def test_refinement_adds_validated_steps_with_one_llm_call():
    c = ctx([refine_json([extra_step()])])
    p = Planner(refine=True).plan("r", decision("general_research"), routed(), c)
    assert "extra1" in [s.id for s in p.steps] and c.budget.usage()["llm_calls"] == 1
    validate_plan(p, BIG, TOOLS)
    assert p.steps[-1].agent == "synthesis_agent"
    assert "extra1" in p.steps[-1].depends_on or any("extra1" in s.depends_on for s in p.steps)


def test_refinement_with_invalid_output_or_disallowed_tool_is_ignored():
    c = ctx(["garbage"] * 3)
    p = Planner(refine=True).plan("r", decision("general_research"), routed(), c)
    assert [s.id for s in p.steps] == ["search", "read", "extract", "verify", "synthesize"]
    assert any("refinement ignored" in n for n in p.notes)
    c2 = ctx([refine_json([extra_step(tools=("teleport",))])])
    p2 = Planner(refine=True).plan("r", decision("general_research"), routed(), c2)
    assert "extra1" not in [s.id for s in p2.steps] and any("refinement ignored" in n for n in p2.notes)


def test_refinement_is_capped_at_max_extra_steps():
    c = ctx([refine_json([extra_step(f"more{i}") for i in range(10)])])
    p = Planner(refine=True).plan("r", decision("general_research"), routed(), c)
    assert len([s for s in p.steps if s.id.startswith("more")]) == 2
