import pytest

from research_os.contracts import ExecutionBudget
from research_os.graph.templates import AGENTS, budget_for, get_workflow, load_workflows
from research_os.prompts import load_prompt
from research_os.router.workflows import workflow_map

INTENTS = ["FACT_LOOKUP", "TECHNOLOGY_COMPARISON", "LITERATURE_REVIEW", "TECHNICAL_DEEP_DIVE",
           "COMPETITIVE_INTELLIGENCE", "MARKET_RESEARCH", "RESEARCH_MONITORING", "GENERAL_RESEARCH"]


def test_every_intent_resolves_to_a_complete_template():
    wf = load_workflows()
    assert set(workflow_map()) == set(INTENTS)
    for name, w in wf.items():
        assert w["steps"], name
        assert w["steps"][-1]["agent"] == "synthesis_agent", f"{name}: last step must be synthesis"
        assert w["budget_profile"] in ("small", "medium", "large")
        assert set(w["required_source_types"]) <= set(w["allowed_source_types"])


def test_step_templates_reference_valid_agents_and_dependencies():
    for name, w in load_workflows().items():
        ids = [s["id"] for s in w["steps"]]
        assert len(ids) == len(set(ids)), name
        for i, s in enumerate(w["steps"]):
            assert s["agent"] in AGENTS, f"{name}:{s['id']}"
            assert all(d in ids[:i] for d in s.get("depends_on", [])), f"{name}:{s['id']} depends on a later/unknown step"


def test_get_workflow_unknown_raises():
    with pytest.raises(KeyError):
        get_workflow("nope")


def test_budget_profiles_are_valid_contracts_and_scale():
    small, large = budget_for("fact_lookup"), budget_for("literature_review")
    assert isinstance(small, ExecutionBudget)
    assert large.max_searches > small.max_searches and large.max_llm_calls > small.max_llm_calls


def test_market_research_requires_dated_sources_and_monitoring_is_dated_too():
    assert get_workflow("market_research")["require_dates"] is True
    assert get_workflow("research_monitoring")["require_dates"] is True
    assert get_workflow("fact_lookup").get("require_dates", False) is False


def test_prompt_loader_versions_and_renders():
    p = load_prompt("supervisor")
    assert p.name == "supervisor" and p.version.startswith("v") and p.id == f"supervisor.{p.version}"
    out = p.render(request="compare a and b", workflows="- x: y", guess="general_research", confidence="0.40")
    assert "compare a and b" in out and "{request}" not in out


def test_prompt_loader_picks_highest_version_and_rejects_missing_vars(tmp_path):
    (tmp_path / "t.v1.md").write_text("one {a}")
    (tmp_path / "t.v2.md").write_text("two {a}")
    p = load_prompt("t", directory=str(tmp_path))
    assert p.version == "v2" and p.render(a="x") == "two x"
    with pytest.raises(KeyError):
        p.render()
    with pytest.raises(FileNotFoundError):
        load_prompt("missing", directory=str(tmp_path))
    assert load_prompt("t", version="v1", directory=str(tmp_path)).render(a="y") == "one y"
