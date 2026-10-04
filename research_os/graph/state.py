"""LangGraph state for the research workflow."""
from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Flows through the graph. Later milestones add claims, verification and report."""

    request: str
    router_output: Any   # RouterOutput
    decision: Any        # SupervisorDecision
    plan: Any            # ResearchPlan
    execution: Any       # ExecutionResult
    status: str          # RunStatus value
    error: str | None
    partial_artifact: dict | None  # set when the run ends PARTIAL / FAILED
    recall: list | None            # memory hits when the request asked about earlier research
    prior_findings: list           # fresh, verifiable claims from earlier runs offered to the planner
    monitoring_diff: dict | None   # new vs unchanged findings for research_monitoring runs


def initial_state(request: str) -> ResearchState:
    """Fresh state for one request."""
    return {"request": request, "status": "RUNNING", "error": None}
