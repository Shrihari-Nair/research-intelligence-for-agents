"""Human-in-the-loop approval policy: when a plan (or a finished report) must wait for a person."""
from __future__ import annotations

from dataclasses import dataclass

from research_os.common import load_config
from research_os.contracts import CostEstimate

# State types that LangGraph's checkpointer must be allowed to (de)serialize when a run pauses at an interrupt.
_ALLOWED = [("research_os.contracts.core", n) for n in ("ResearchPlan", "ResearchStep", "CostEstimate", "RunStatus", "Source", "Claim")] + [
    ("research_os.contracts.router", "RouterOutput"), ("research_os.contracts.router", "RoutingMode"),
    ("research_os.agents.supervisor", "SupervisorDecision"), ("research_os.graph.executor", "ExecutionResult")]


@dataclass(frozen=True)
class ApprovalPolicy:
    """Thresholds from configs/budgets.yaml (`approvals:`)."""

    max_searches: int
    max_sources: int
    max_llm_calls: int
    publish: bool
    expiry_seconds: float

    @classmethod
    def from_config(cls, path: str = "configs/budgets.yaml") -> ApprovalPolicy:
        c = load_config(path)["approvals"]
        return cls(c["max_searches"], c["max_sources"], c["max_llm_calls"], bool(c["publish"]), float(c["expiry_seconds"]))


def needs_plan_approval(cost: CostEstimate, policy: ApprovalPolicy) -> list[str]:
    """Human-readable reasons the plan needs approval (empty = run without asking). A value AT the threshold does not trigger."""
    out = []
    if cost.searches > policy.max_searches:
        out.append(f"{cost.searches} searches (threshold {policy.max_searches})")
    if cost.sources > policy.max_sources:
        out.append(f"about {cost.sources} sources (threshold {policy.max_sources})")
    if cost.llm_calls > policy.max_llm_calls:
        out.append(f"about {cost.llm_calls} LLM calls (threshold {policy.max_llm_calls})")
    return out


def make_checkpointer():
    """In-memory LangGraph checkpointer that explicitly allows our state types (needed for pause/resume across API calls)."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED))
