"""API request/response models (the OpenAPI schema the frontend types are generated from)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_os.contracts import RunStatus


class CreateRun(BaseModel):
    """POST /runs body."""

    text: str = Field(min_length=1, description="The research request")
    budget_profile: Literal["small", "medium", "large"] | None = Field(default=None, description="Override the workflow's budget profile")


class RejectBody(BaseModel):
    """POST /runs/{id}/reject body."""

    reason: str | None = None


class RunCreated(BaseModel):
    """POST /runs response."""

    run_id: str


class StepView(BaseModel):
    """One plan step with live status."""

    id: str
    agent: str
    objective: str
    status: Literal["pending", "running", "succeeded", "failed", "skipped"]
    attempts: int = 0
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """One row of run history."""

    run_id: str
    request: str
    status: RunStatus
    workflow: str | None
    created_at: datetime
    duration_s: float | None
    n_sources: int
    n_supported_claims: int
    n_conflicts: int


class RunDetail(BaseModel):
    """GET /runs/{id}."""

    run_id: str
    request: str
    status: RunStatus
    workflow: str | None
    llm_backend: str
    created_at: datetime
    finished_at: datetime | None
    router: dict[str, Any] | None
    decision: dict[str, Any] | None
    plan: list[StepView]
    plan_notes: list[str] = Field(default_factory=list)
    budget: dict[str, Any]
    claims: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    recoveries: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    partial: dict[str, Any] | None
    error: str | None
    trace_url: str | None = None
    approval: dict[str, Any] | None = None
    recall: list[dict[str, Any]] | None = None
    prior_findings: list[dict[str, Any]] = Field(default_factory=list)
    monitoring_diff: dict[str, Any] | None = None


class ReportResponse(BaseModel):
    """GET /runs/{id}/report."""

    run_id: str
    status: RunStatus
    report: dict[str, Any]


class SourceContent(BaseModel):
    """GET /runs/{id}/sources/{source_id}: the stored text a claim's quote must appear in."""

    source_id: str
    title: str
    url: str
    source_type: str
    published_at: str | None = None
    content: str
