"""Failure recovery: classify -> policy table -> bounded actions. Recovery can never loop forever (per-step and per-run caps)."""
from __future__ import annotations

from dataclasses import dataclass

from research_os.common import load_config
from research_os.contracts import FailureType, ResearchStep
from research_os.tools import SEARCH_TOOLS

RERUN_ACTIONS = ("retry", "retry_with_error", "broaden_query", "alternate_tool", "skip_source")
TERMINAL_ACTIONS = ("terminate_step", "skip_step", "terminate_safely", "ask_user", "replan", "proceed")


@dataclass
class RecoveryDecision:
    """What recovery decided for one failure."""

    action: str
    failure_type: FailureType
    step_id: str
    attempt: int
    reason: str = ""


class RecoveryManager:
    """Stateful per run: tracks recoveries per step and per run, and replans used."""

    def __init__(self, policy: dict | None = None, path: str = "configs/recovery.yaml"):
        cfg = policy if policy is not None else load_config(path)
        self.limits, self.alternates, self.policy = cfg["limits"], cfg.get("alternates", {}), cfg.get("policy", {})
        self.step_idx: dict[str, int] = {}
        self.step_count: dict[str, int] = {}
        self.run_count = 0
        self.replans = 0

    # ------------------------------------------------------------------
    def _alternatives(self, step: ResearchStep) -> dict[str, str]:
        tried = set(step.params.get("tools_tried", [])) | set(step.tools)
        out = {}
        for tool in step.tools:
            if tool in SEARCH_TOOLS or tool == "fetch_url":
                alt = next((a for a in self.alternates.get(tool, []) if a not in tried), None)
                if alt:
                    out[tool] = alt
                    tried.add(alt)
        return out

    def _applicable(self, action: str, step: ResearchStep) -> bool:
        if action == "alternate_tool":
            return bool(self._alternatives(step))
        if action == "broaden_query":
            return step.agent == "search_agent"
        if action == "skip_source":
            return step.agent in ("search_agent", "source_agent")
        if action == "replan":
            return self.replans < int(self.limits.get("max_replans", 1))
        return True

    def decide(self, step: ResearchStep, failure_type: FailureType, ctx, exc: BaseException | None = None) -> RecoveryDecision:
        """Choose the next action for this failure and emit a `recovery` event."""
        sid = step.id
        if failure_type == FailureType.BUDGET_EXHAUSTED:  # a spent budget is a hard stop, whatever the policy file says
            d = RecoveryDecision("terminate_safely", failure_type, sid, self.step_idx.get(sid, 0) + 1, "budget exhausted")
        elif self.run_count >= self.limits["max_per_run"]:
            d = RecoveryDecision("terminate_safely", failure_type, sid, self.step_idx.get(sid, 0) + 1, "per-run recovery cap reached")
        elif self.step_count.get(sid, 0) >= self.limits["max_per_step"]:
            d = RecoveryDecision("terminate_step", failure_type, sid, self.step_idx.get(sid, 0) + 1, "per-step recovery cap reached")
        else:
            seq = self.policy.get(failure_type.value, [])
            idx = self.step_idx.get(sid, 0)
            action = None
            while idx < len(seq):
                cand = seq[idx]
                idx += 1
                if self._applicable(cand, step):
                    action = cand
                    break
            self.step_idx[sid] = idx
            d = RecoveryDecision(action or "terminate_step", failure_type, sid, idx,
                                 "" if action else "no further recovery actions for this failure")
        self.run_count += 1
        self.step_count[sid] = self.step_count.get(sid, 0) + 1
        if d.action == "replan":
            self.replans += 1
        ctx.events.emit("recovery", "recovery", {"step": sid, "failure_type": failure_type.value, "action": d.action,
                                                 "attempt": d.attempt, "reason": d.reason, "error": str(exc)[:200] if exc else ""})
        return d

    def apply(self, d: RecoveryDecision, step: ResearchStep, exc: BaseException | None) -> None:
        """Mutate the step so its re-run differs from the failed attempt."""
        p = step.params
        if d.action == "retry_with_error":
            p["last_error"] = str(exc)[:300] if exc else "unknown error"
        elif d.action == "broaden_query":
            p["broaden_level"] = int(p.get("broaden_level", 0)) + 1
        elif d.action == "alternate_tool":
            swaps = self._alternatives(step)
            p["tools_tried"] = sorted(set(p.get("tools_tried", [])) | set(step.tools) | set(swaps.values()))
            step.tools = [swaps.get(t, t) for t in step.tools]
        elif d.action == "skip_source":
            sid = getattr(exc, "source_id", None)
            if sid:
                p["skip_source_ids"] = [*p.get("skip_source_ids", []), sid]


def build_partial_artifact(result, ctx, question: str) -> dict:
    """A valid PARTIAL artifact stating what was and was not completed (always producible, even when nothing completed)."""
    plan = result.plan
    fail_by_step = {f["step"]: f["failure_type"] for f in result.failures}
    claims, conflicts, unverified = [], [], 0
    for out in result.outputs.values():
        for c in out.get("claims", []):
            if c.get("verdict") == "SUPPORTED":
                claims.append(c)
            else:
                unverified += 1
        conflicts.extend(out.get("conflicts", []))
    done = [s.id for s in plan.steps if s.status == "succeeded"]
    todo = [{"id": s.id, "status": s.status, "failure_type": fail_by_step.get(s.id)} for s in plan.steps if s.status != "succeeded"]
    if result.ask_user:
        message = result.ask_user
    elif result.halted_reason:
        message = f"The run stopped early ({result.halted_reason}). {len(done)} of {len(plan.steps)} steps completed."
    else:
        message = f"{len(done)} of {len(plan.steps)} steps completed; the rest failed or were skipped."
    return {"status": "PARTIAL", "question": question, "message": message, "halted_reason": result.halted_reason or "",
            "completed_steps": done, "incomplete_steps": todo, "claims": claims, "claims_not_supported": unverified, "conflicts": conflicts,
            "sources": [{"source_id": s.source_id, "url": s.url, "title": s.title} for s in ctx.sources.all()],
            "recoveries": result.recoveries, "ask_user": result.ask_user, "budget": ctx.budget.snapshot()}


def empty_partial(question: str, reason: str, ctx, message: str | None = None) -> dict:
    """PARTIAL artifact for a run that never produced a plan to execute."""
    return {"status": "PARTIAL", "question": question, "message": message or f"The run stopped before any step ran ({reason}).",
            "halted_reason": reason, "completed_steps": [], "incomplete_steps": [], "claims": [], "claims_not_supported": 0,
            "conflicts": [], "sources": [{"source_id": s.source_id, "url": s.url, "title": s.title} for s in ctx.sources.all()],
            "recoveries": [], "ask_user": None, "budget": ctx.budget.snapshot()}
