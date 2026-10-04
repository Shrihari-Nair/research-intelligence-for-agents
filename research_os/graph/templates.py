"""Workflow templates and budget profiles from configs/workflows.yaml and configs/budgets.yaml."""
from __future__ import annotations

from research_os.common import load_config
from research_os.contracts import ExecutionBudget

AGENTS = ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")


def load_workflows(path: str = "configs/workflows.yaml") -> dict[str, dict]:
    """All workflow specs keyed by workflow name."""
    return load_config(path)["workflows"]


def planner_cfg(path: str = "configs/workflows.yaml") -> dict:
    """Planner settings (refine flag, extra step cap, cost model)."""
    return load_config(path)["planner"]


def get_workflow(name: str, path: str = "configs/workflows.yaml") -> dict:
    """One workflow spec; KeyError if unknown."""
    wf = load_workflows(path)
    if name not in wf:
        raise KeyError(f"unknown workflow {name!r}; known: {sorted(wf)}")
    return wf[name]


def budget_for(workflow: str, workflows_path: str = "configs/workflows.yaml",
               budgets_path: str = "configs/budgets.yaml") -> ExecutionBudget:
    """ExecutionBudget of the workflow's profile."""
    profile = get_workflow(workflow, workflows_path)["budget_profile"]
    return ExecutionBudget(**load_config(budgets_path)["profiles"][profile])
