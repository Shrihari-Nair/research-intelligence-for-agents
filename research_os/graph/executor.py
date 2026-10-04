"""Executor: runs plan steps in dependency order through agents. Failures never raise; they go to recovery (if configured)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from research_os.contracts import FailureType, ResearchPlan, ResearchStep, RunStatus
from research_os.llm.structured import MalformedLLMOutput
from research_os.tools.registry import ToolError

ASK_USER_MESSAGE = ("I could not find useful, verifiable sources for this request. Please rephrase it or narrow it down "
                    "(for example name the specific technology, company or time period).")


@dataclass
class ExecutionResult:
    """What happened while running a plan."""

    plan: ResearchPlan
    status: RunStatus
    outputs: dict[str, dict] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    recoveries: list[dict] = field(default_factory=list)
    halted_reason: str | None = None
    ask_user: str | None = None
    replan_requested: bool = False


def merge_history(prev: ExecutionResult | None, new: ExecutionResult) -> ExecutionResult:
    """Carry an earlier execution's failures and recoveries (incl. the replan decision) into the replanned one."""
    if prev is not None:
        new.failures = [*prev.failures, *new.failures]
        new.recoveries = [*prev.recoveries, *new.recoveries]
    return new


class GateFailed(Exception):
    """The verifier's gate returned FAIL. Carries the verification outputs so they survive into the partial artifact."""

    def __init__(self, outputs: dict):
        gate = outputs["gate"]
        super().__init__(f"verification gate FAIL: {gate['reason']}")
        self.outputs = outputs
        self.failure_type = FailureType(gate["failure_type"])


def classify_failure(exc: BaseException) -> FailureType:
    """Map any exception to a FailureType (recovery policies key on this)."""
    if isinstance(exc, ToolError):
        return exc.failure_type
    if isinstance(exc, MalformedLLMOutput):
        return FailureType.MALFORMED_LLM_OUTPUT
    ft = getattr(exc, "failure_type", None)
    return ft if isinstance(ft, FailureType) else FailureType.AGENT_FAILURE


def _topological(steps: list[ResearchStep]) -> list[ResearchStep]:
    done: set[str] = set()
    order: list[ResearchStep] = []
    pending = list(steps)
    while pending:
        progressed = False
        for s in list(pending):
            if all(d in done for d in s.depends_on):
                order.append(s)
                done.add(s.id)
                pending.remove(s)
                progressed = True
        if not progressed:  # cyclic / dangling: validate_plan prevents this; do not loop forever
            order.extend(pending)
            break
    return order


def execute_plan(plan: ResearchPlan, agents: dict, ctx, request: str, on_failure: Callable[[ResearchStep, BaseException], None] | None = None,
                 recovery=None) -> ExecutionResult:
    """Run every step (re-running per recovery decisions). Dependents of a failed/skipped step are skipped."""
    outputs: dict[str, dict] = {}
    failures: list[dict] = []
    recoveries: list[dict] = []
    status_of: dict[str, str] = {}
    res = ExecutionResult(plan=plan, status=RunStatus.RUNNING)
    halted = False

    def skip(step, why):
        step.status = "skipped"
        status_of[step.id] = "skipped"
        ctx.events.emit(f"step:{step.id}", "info", {"skipped": why})

    for step in _topological(plan.steps):
        node = f"step:{step.id}"
        if halted:
            skip(step, "run halted")
            continue
        if any(status_of.get(d) != "succeeded" for d in step.depends_on):
            skip(step, "dependency did not succeed")
            continue
        agent = agents.get(step.agent)
        while True:  # attempt loop; each re-run is a recovery decision and consumes budget
            step.attempts += 1
            exc: BaseException | None = None
            out: dict = {}
            try:
                ctx.budget.count("steps")
                step.status = "running"
                ctx.events.emit(node, "started", {"agent": step.agent, "objective": step.objective, "attempt": step.attempts})
                if agent is None:
                    raise LookupError(f"no agent registered for {step.agent!r}")
                out = agent.run(step, {d: outputs[d] for d in step.depends_on}, ctx, request)
                gate = out.get("gate") if isinstance(out, dict) else None
                if gate and gate.get("result") == "FAIL":
                    raise GateFailed(out)
            except Exception as e:  # noqa: BLE001 - every failure is classified, never propagated
                exc = e
            if exc is None or isinstance(exc, GateFailed):
                for fn in getattr(ctx, "step_listeners", []):  # incremental persistence: claims survive even if the run later crashes
                    try:
                        fn(step, exc.outputs if isinstance(exc, GateFailed) else out)
                    except Exception:  # noqa: BLE001 - a listener must never break a run
                        pass
            if exc is None:
                step.status, step.outputs = "succeeded", out
                status_of[step.id] = "succeeded"
                outputs[step.id] = out
                ctx.events.emit(node, "finished", {"output_keys": sorted(out)})
                break
            ft = classify_failure(exc)
            if isinstance(exc, GateFailed):
                out = exc.outputs
            failures.append({"step": step.id, "agent": step.agent, "failure_type": ft.value, "error": f"{type(exc).__name__}: {exc}"})
            ctx.events.emit(node, "failed", {"failure_type": ft.value, "error": f"{type(exc).__name__}: {exc}"})
            if on_failure:
                on_failure(step, exc)
            if recovery is None:
                halted = halted or ft == FailureType.BUDGET_EXHAUSTED
                step.status, status_of[step.id] = "failed", "failed"
                if isinstance(exc, GateFailed):
                    step.outputs, outputs[step.id] = out, out
                break
            d = recovery.decide(step, ft, ctx, exc)
            recoveries.append({"step": step.id, "failure_type": ft.value, "action": d.action, "attempt": d.attempt, "reason": d.reason,
                               "outcome": "re-running step" if d.action in ("retry", "retry_with_error", "broaden_query", "alternate_tool",
                                                                          "skip_source") else "step finalized"})
            if d.action in ("retry", "retry_with_error", "broaden_query", "alternate_tool", "skip_source"):
                recovery.apply(d, step, exc)
                continue
            if d.action == "skip_step":
                skip(step, "recovery: skip_step")
            elif d.action == "proceed" and isinstance(exc, GateFailed):
                step.status, step.outputs = "succeeded", out
                status_of[step.id], outputs[step.id] = "succeeded", out
            else:
                step.status, status_of[step.id] = "failed", "failed"
                if isinstance(exc, GateFailed):
                    step.outputs, outputs[step.id] = out, out
            if d.action == "terminate_safely":
                halted = True
                res.halted_reason = (f"budget exhausted: {exc}" if ft == FailureType.BUDGET_EXHAUSTED else d.reason or "terminated safely")
            elif d.action == "ask_user":
                halted, res.ask_user, res.halted_reason = True, ASK_USER_MESSAGE, "no useful sources; user input requested"
            elif d.action == "replan":
                halted, res.replan_requested = True, True
            break
    n_ok = sum(s.status == "succeeded" for s in plan.steps)
    stopped = halted and not res.replan_requested
    if halted and not res.halted_reason and not res.replan_requested:
        res.halted_reason = "budget exhausted"
    res.status = RunStatus.SUCCESS if n_ok == len(plan.steps) else RunStatus.PARTIAL if (n_ok or stopped) else RunStatus.FAILED
    res.outputs, res.failures, res.recoveries = outputs, failures, recoveries
    return res
