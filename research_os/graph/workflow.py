"""START -> router -> supervisor -> planner -> executor -> END, with recovery loops (replan) and partial artifacts."""
from __future__ import annotations

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from research_os.agents.planner import Planner, PlanValidationError
from research_os.contracts import ExecutionBudget, FailureType, ResearchStep
from research_os.graph.executor import execute_plan, merge_history
from research_os.graph.state import ResearchState, initial_state
from research_os.graph.templates import budget_for
from research_os.harness.approvals import ApprovalPolicy, make_checkpointer, needs_plan_approval
from research_os.harness.recovery import RecoveryManager, build_partial_artifact, empty_partial
from research_os.memory import memory_cfg

__all__ = ["build_graph", "initial_state"]
_DEFAULT = object()
_ORDER = ["router", "supervisor", "planner", "executor"]


def build_graph(router, supervisor, ctx, *, agents: dict | None = None, planner: Planner | None = None,
                recovery: RecoveryManager | None | object = _DEFAULT, stop_after: str | None = None,
                approvals: ApprovalPolicy | None = None, checkpointer=None):
    """Compile the research graph. `stop_after` ends the graph after that node (used by early-milestone tests)."""
    planner = planner or Planner()
    recovery = RecoveryManager() if recovery is _DEFAULT else recovery  # None = recovery disabled (ablations / minimal harness)

    def router_node(state: ResearchState) -> dict:
        ctx.events.emit("router", "started", {})
        out = router.route(state["request"])
        ctx.events.emit("router", "finished", {"intent": out.intent, "confidence": out.confidence,
                                               "mode": out.routing_mode.value, "latency_ms": out.latency_ms})
        return {"router_output": out}

    def supervisor_node(state: ResearchState) -> dict:
        return {"decision": supervisor.decide(state["request"], state["router_output"], ctx)}

    def planner_node(state: ResearchState) -> dict:
        override = ctx.config.get("budget")
        ctx.budget.set_limits(ExecutionBudget(**override) if override else budget_for(state["decision"].workflow))
        ctx.events.emit("budget", "info", {"limits": ctx.budget.snapshot()["limits"]})
        args = (state["request"], state["decision"], state["router_output"], ctx)
        prior: list = []
        if ctx.memory is not None:
            reuse = memory_cfg()["reuse"]
            try:
                prior = ctx.memory.prior_findings(state["request"], now=datetime.now(timezone.utc), max_age_days=reuse["max_age_days"],
                                                  min_similarity=reuse["min_similarity"], k=reuse["max_prior_findings"])
            except Exception:  # noqa: BLE001 - memory is an optimization; research proceeds without it
                prior = []
        try:
            plan = planner.plan(*args)
            if prior:
                plan.notes.append(f"{len(prior)} prior finding(s) available from earlier runs (fresh, source and quote still stored)")
            return {"plan": plan, "prior_findings": prior}
        except PlanValidationError as e:
            d = recovery.decide(ResearchStep(id="plan", objective="plan", agent="planner", tools=[], depends_on=[]),
                                FailureType.IMPOSSIBLE_PLAN, ctx, e) if recovery is not None else None
            if d is not None and d.action == "replan":
                try:
                    return {"plan": planner.replan(*args, FailureType.IMPOSSIBLE_PLAN)}
                except PlanValidationError as e2:
                    e = e2
                    recovery.decide(ResearchStep(id="plan", objective="plan", agent="planner", tools=[], depends_on=[]),
                                    FailureType.IMPOSSIBLE_PLAN, ctx, e2)
            reason = f"no valid plan could be built ({e})"
            return {"status": "PARTIAL", "error": reason,
                    "partial_artifact": empty_partial(state["request"], reason, ctx, f"No valid research plan could be built: {e}")}

    def recall_node(state: ResearchState) -> dict:
        ctx.events.emit("recall", "finished", {"hits": len(state["decision"].recall), "from": [h["run_id"] for h in state["decision"].recall]})
        return {"recall": state["decision"].recall, "status": "SUCCESS"}

    def after_supervisor(state: ResearchState) -> str:
        return "recall" if state["decision"].recall else "planner"

    def approval_node(state: ResearchState) -> dict:
        """Pause (LangGraph interrupt) when the plan is expensive. The node is re-run from its start on resume, so it must stay pure:
        events about the request/decision are emitted by the service that observes the interrupt."""
        plan = state["plan"]
        reasons = needs_plan_approval(plan.estimated_cost, approvals)
        if not reasons:
            return {}
        limits = ctx.budget.limits()
        answer = interrupt({"kind": "plan", "workflow": plan.workflow, "searches": plan.estimated_cost.searches,
                            "expected_sources": plan.estimated_cost.sources, "llm_calls": plan.estimated_cost.llm_calls,
                            "tool_calls": plan.estimated_cost.tool_calls, "budget": limits.model_dump() if limits else None,
                            "reasons": reasons, "steps": len(plan.steps)})
        if answer.get("decision") != "approve":
            return {"status": "CANCELLED", "error": f"plan rejected: {answer.get('reason') or 'no reason given'}"}
        return {}

    def after_approval(state: ResearchState) -> str:
        return END if state.get("status") == "CANCELLED" else "executor"

    def publish_node(state: ResearchState) -> dict:
        synth = next((s.outputs for s in state["plan"].steps if s.agent == "synthesis_agent" and s.outputs), {}) if state.get("plan") else {}
        answer = interrupt({"kind": "publish", "status": state.get("status"), "summary": synth.get("summary", ""),
                            "findings": len(synth.get("findings", []))})
        if answer.get("decision") != "approve":
            return {"status": "CANCELLED", "error": f"publication rejected: {answer.get('reason') or 'no reason given'}"}
        return {}

    def after_planner(state: ResearchState) -> str:
        if state.get("plan") is None:
            return END
        return "approval" if approvals is not None else "executor"

    def run_plan(state: ResearchState, plan, prev=None) -> dict:
        if agents is None:
            raise RuntimeError("build_graph needs `agents` to execute a plan")
        res = merge_history(prev, execute_plan(plan, agents, ctx, state["request"], recovery=recovery))
        out: dict = {"plan": res.plan, "execution": res, "status": res.status.value}
        if ctx.memory is not None and state["decision"].workflow == "research_monitoring" and not res.replan_requested:
            verify = next((s.outputs for s in res.plan.steps if s.agent == "verifier"), {})
            supported = [c for c in verify.get("claims", []) if c.get("verdict") == "SUPPORTED"]
            try:
                out["monitoring_diff"] = ctx.memory.diff_with_prior(state["request"], supported, exclude_run_id=ctx.run_id,
                                                                    similarity=memory_cfg()["monitoring"]["same_claim_similarity"])
            except Exception:  # noqa: BLE001
                out["monitoring_diff"] = None
        if res.status.value != "SUCCESS" and not res.replan_requested:
            out["partial_artifact"] = build_partial_artifact(res, ctx, state["request"])
        return out

    def executor_node(state: ResearchState) -> dict:
        return run_plan(state, state["plan"])

    def after_executor(state: ResearchState) -> str:
        if state["execution"].replan_requested:
            return "replanner"
        return "publish" if approvals is not None and approvals.publish and state.get("status") in ("SUCCESS", "PARTIAL") else END

    def replanner_node(state: ResearchState) -> dict:
        res = state["execution"]
        reason = FailureType(res.failures[-1]["failure_type"]) if res.failures else FailureType.NO_USEFUL_SOURCES
        try:
            new_plan = planner.replan(state["request"], state["decision"], state["router_output"], ctx, reason)
        except PlanValidationError as e:
            msg = f"replanning failed ({e})"
            return {"status": "PARTIAL", "error": msg, "partial_artifact": build_partial_artifact(res, ctx, state["request"])}
        return run_plan(state, new_plan, prev=res)

    nodes = {"router": router_node, "supervisor": supervisor_node, "planner": planner_node, "executor": executor_node}
    last = _ORDER.index(stop_after) if stop_after else len(_ORDER) - 1
    g = StateGraph(ResearchState)
    for name in _ORDER[: last + 1]:
        g.add_node(name, nodes[name])
    g.add_edge(START, "router")
    for a, b in zip(_ORDER[:last], _ORDER[1: last + 1], strict=True):
        if a == "supervisor":
            g.add_node("recall", recall_node)
            g.add_conditional_edges("supervisor", after_supervisor, {"recall": "recall", "planner": "planner"})
            g.add_edge("recall", END)
        elif a == "planner":
            g.add_conditional_edges("planner", after_planner, {"executor": "executor", "approval": "approval", END: END})
        else:
            g.add_edge(a, b)
    if last == len(_ORDER) - 1:
        g.add_node("replanner", replanner_node)
        g.add_node("approval", approval_node)
        g.add_node("publish", publish_node)
        g.add_conditional_edges("approval", after_approval, {"executor": "executor", END: END})
        g.add_conditional_edges("executor", after_executor, {"replanner": "replanner", "publish": "publish", END: END})
        g.add_conditional_edges("replanner", after_executor, {"replanner": "replanner", "publish": "publish", END: END})
        g.add_edge("publish", END)
    else:
        g.add_edge(_ORDER[last], END)
    if approvals is not None and checkpointer is None:
        checkpointer = make_checkpointer()
    return g.compile(checkpointer=checkpointer)
