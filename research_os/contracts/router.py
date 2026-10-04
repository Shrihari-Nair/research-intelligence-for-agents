"""Router contracts."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RoutingMode(str, Enum):
    """DIRECT: router confident, supervisor needs no LLM. LLM_FALLBACK: supervisor asks the LLM."""

    DIRECT = "DIRECT"
    LLM_FALLBACK = "LLM_FALLBACK"


class RouterOutput(BaseModel):
    """What the MiniLM router hands the supervisor.

    `workflow` is the workflow for the router's own predicted intent; in LLM_FALLBACK mode the supervisor
    may override it. `entities` always has technologies/companies/topics/time_scope (lists of str).
    """

    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    routing_mode: RoutingMode
    workflow: str
    entities: dict[str, list[str]]
    latency_ms: float = Field(ge=0.0)
