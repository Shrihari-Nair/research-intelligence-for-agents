"""Core research contracts (section 4 of ROADMAP2). Keep stable: API, frontend, evaluation and replay depend on them."""
from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator

_TRACKING = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref_src")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonical_url(url: str) -> str:
    """Lowercase scheme/host, drop default ports, fragments and tracking params, sort query, ensure a path."""
    p = urlsplit(url.strip())
    scheme, host = p.scheme.lower(), (p.hostname or "").lower()
    netloc = host if p.port in (None, _DEFAULT_PORTS.get(scheme)) else f"{host}:{p.port}"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                             if not k.lower().startswith(_TRACKING)))
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, query, ""))


def source_id_for_url(url: str) -> str:
    """Stable 16-hex-char id: hash of the canonical URL."""
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:16]


class Source(BaseModel):
    """A fetched/retrieved source with cleaned, size-capped text. Content is untrusted data."""

    source_id: str
    title: str
    url: str
    source_type: Literal["documentation", "paper", "repository", "news", "blog", "forum", "other"]
    published_at: datetime | None = None
    retrieved_at: datetime
    content: str
    tool: str

    @field_validator("url")
    @classmethod
    def _web_url_only(cls, v: str) -> str:
        if urlsplit(v.strip()).scheme.lower() not in ("http", "https"):
            raise ValueError("source url must be http(s)")
        return v


class Claim(BaseModel):
    """One atomic claim grounded in a verbatim quote from a stored source."""

    claim_id: str
    claim: str = Field(min_length=1)
    source_id: str
    evidence_quote: str = Field(min_length=1)
    extractor_confidence: float = Field(ge=0.0, le=1.0)
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIED"] = "UNVERIFIED"
    verdict_rationale: str | None = None


class CostEstimate(BaseModel):
    """Plan-derived cost estimate (feeds approvals in M22)."""

    searches: int = 0
    sources: int = 0
    llm_calls: int = 0
    tool_calls: int = 0


class ResearchStep(BaseModel):
    """One step of a research plan."""

    id: str
    objective: str
    agent: str
    tools: list[str]
    depends_on: list[str]
    status: Literal["pending", "running", "succeeded", "failed", "skipped"] = "pending"
    attempts: int = 0
    outputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)  # entity, request, recent, ... set by the planner


class ResearchPlan(BaseModel):
    """Ordered steps with dependencies for one workflow."""

    plan_id: str
    workflow: str
    steps: list[ResearchStep]
    estimated_cost: CostEstimate
    notes: list[str] = Field(default_factory=list)


class ExecutionBudget(BaseModel):
    """Hard limits for one run. Every limit must be positive."""

    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_searches: int = Field(gt=0)
    max_llm_calls: int = Field(gt=0)
    max_runtime_seconds: int = Field(gt=0)


class RunEvent(BaseModel):
    """One observable thing that happened in a run."""

    run_id: str
    seq: int = Field(ge=0)
    ts: datetime
    node: str
    type: Literal["started", "finished", "failed", "retry", "recovery", "budget", "approval", "info"]
    payload: dict[str, Any] = Field(default_factory=dict)


class RunStatus(str, Enum):
    """Lifecycle of a run."""

    PENDING = "PENDING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FailureType(str, Enum):
    """Failure classes recovery policies are keyed on."""

    SEARCH_TIMEOUT = "SEARCH_TIMEOUT"
    INVALID_SOURCE = "INVALID_SOURCE"
    MALFORMED_LLM_OUTPUT = "MALFORMED_LLM_OUTPUT"
    NO_USEFUL_SOURCES = "NO_USEFUL_SOURCES"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    TOOL_FAILURE = "TOOL_FAILURE"
    AGENT_FAILURE = "AGENT_FAILURE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    IMPOSSIBLE_PLAN = "IMPOSSIBLE_PLAN"
