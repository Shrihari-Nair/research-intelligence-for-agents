"""The four compared systems (+ harness ablations).

A  LLM-only routing   : the LLM decides the workflow for every request; minimal harness
B  MiniLM-only        : the classifier always decides (never asks the LLM); minimal harness
C  Hybrid             : threshold routing (LLM only when unsure); minimal harness
D  Hybrid + Harness   : C plus budgets, verifier gate, recovery

"Minimal harness" = no verifier (claims pass straight to synthesis, nothing is gated), no recovery, and only a global step cap.
Entity extraction (deterministic) is shared by all systems. Ablations remove one component from D.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from research_os.agents import build_agents
from research_os.agents.planner import Planner
from research_os.agents.supervisor import Supervisor
from research_os.agents.synthesis_agent import merged_claims
from research_os.contracts import ExecutionBudget, RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.recovery import RecoveryManager

SYSTEMS = {
    "A": "LLM-only routing, minimal harness",
    "B": "MiniLM-only routing, minimal harness",
    "C": "Hybrid routing, minimal harness",
    "D": "Hybrid routing + full harness",
    "D-no-verifier": "D without the verifier (claims pass ungated)",
    "D-no-recovery": "D without failure recovery",
    "D-no-budgets": "D without execution budgets",
}
_HUGE = 10_000_000
MINIMAL_BUDGET = ExecutionBudget(max_steps=60, max_tool_calls=_HUGE, max_searches=_HUGE, max_llm_calls=_HUGE, max_runtime_seconds=_HUGE)
NO_BUDGET = ExecutionBudget(max_steps=_HUGE, max_tool_calls=_HUGE, max_searches=_HUGE, max_llm_calls=_HUGE, max_runtime_seconds=_HUGE)


class PassThroughVerifier:
    """Minimal-harness stand-in: marks every claim SUPPORTED and never fails the gate (nothing is checked)."""

    name = "verifier"

    def run(self, step, inputs, ctx, request):
        claims = [dict(c, verdict="SUPPORTED", verdict_rationale="not verified (minimal harness)", verdict_source="passthrough")
                  for c in merged_claims(inputs)]
        return {"claims": claims, "conflicts": [], "gate": {"result": "PASS", "reason": None, "failure_type": None, "counts": {}},
                "stats": {"total": len(claims), "supported": len(claims), "contradicted": 0, "unsupported": 0,
                          "deterministic_rejections": 0, "judge_calls": 0}}


class LLMOnlyRouter:
    """System A: no classifier. Only the deterministic entity extractor runs; the supervisor must ask the LLM."""

    def __init__(self, entities):
        self.entities = entities

    def route(self, text: str) -> RouterOutput:
        t0 = time.perf_counter()
        ents = self.entities.extract(text)
        return RouterOutput(intent="GENERAL_RESEARCH", confidence=0.0, routing_mode=RoutingMode.LLM_FALLBACK, workflow="general_research",
                            entities=ents, latency_ms=(time.perf_counter() - t0) * 1000)


class ForcedModeRouter:
    """System B: the classifier always decides, whatever its confidence."""

    def __init__(self, inner, mode: RoutingMode = RoutingMode.DIRECT):
        self.inner, self.mode, self.entities = inner, mode, inner.entities

    def route(self, text: str) -> RouterOutput:
        return self.inner.route(text).model_copy(update={"routing_mode": self.mode})


@dataclass
class Pipeline:
    """Everything needed to run one request under a given system."""

    name: str
    router: object
    supervisor: Supervisor
    agents: dict
    planner: Planner
    recovery_factory: Callable[[], RecoveryManager] | None
    budget: ExecutionBudget | None   # None: use each workflow's own budget profile


def build_pipeline(name: str, router, agents: dict | None = None) -> Pipeline:
    """Assemble a named system around a ResearchRouter-like `router` (needs .route and .entities)."""
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}")
    agents = dict(agents or build_agents())
    r = {"A": LLMOnlyRouter(router.entities), "B": ForcedModeRouter(router)}.get(name, router)
    minimal = name in ("A", "B", "C")
    if minimal or name == "D-no-verifier":
        agents["verifier"] = PassThroughVerifier()
    recovery = None if (minimal or name == "D-no-recovery") else RecoveryManager
    budget = MINIMAL_BUDGET if minimal else NO_BUDGET if name == "D-no-budgets" else None
    return Pipeline(name=name, router=r, supervisor=Supervisor(include_guess=(name != "A")), agents=agents, planner=Planner(),
                    recovery_factory=recovery, budget=budget)


def run_pipeline(pipe: Pipeline, request: str, ctx) -> dict:
    """Run one request; returns the final graph state."""
    if pipe.budget is not None:
        ctx.config["budget"] = pipe.budget.model_dump()
    graph = build_graph(pipe.router, pipe.supervisor, ctx, agents=pipe.agents, planner=pipe.planner,
                        recovery=pipe.recovery_factory() if pipe.recovery_factory else None)
    return graph.invoke(initial_state(request))
