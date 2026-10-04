"""RunService: starts runs in background threads and exposes their live state. One RunContext per run."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.types import Command

from research_os.agents.planner import Planner
from research_os.artifacts.report import build_report
from research_os.common import load_config
from research_os.contracts import ExecutionBudget, RunStatus
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.approvals import ApprovalPolicy, make_checkpointer
from research_os.harness.context import RunContext
from research_os.harness.recovery import RecoveryManager
from research_os.memory.hooks import attach_memory, finalize_run
from research_os.observability.tracing import attach_tracing, make_tracer, run_scores

log = logging.getLogger(__name__)
DONE = (RunStatus.SUCCESS, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.CANCELLED)
_ACTIVE_WAIT = RunStatus.AWAITING_APPROVAL


@dataclass
class RunRecord:
    """Everything the API knows about one run."""

    run_id: str
    request: str
    ctx: RunContext
    llm_backend: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING
    state: dict | None = None
    error: str | None = None
    tracer: Any = None
    graph: Any = None
    approval: dict | None = None
    approval_expires: float | None = None
    lock: Any = field(default_factory=threading.RLock)

    @property
    def done(self) -> bool:
        return self.status in DONE

    # ---- views ----
    def _steps_from_events(self) -> list[dict]:
        plan_ev = next((e for e in self.ctx.events.events if e.node == "planner" and e.type == "finished"), None)
        if plan_ev is None:
            return []
        steps = {s["id"]: {**s, "status": "pending", "attempts": 0} for s in plan_ev.payload["plan"]}
        for e in self.ctx.events.events:
            if e.node.startswith("step:") and e.node[5:] in steps:
                st = steps[e.node[5:]]
                if e.type == "started":
                    st["status"], st["attempts"] = "running", st["attempts"] + 1
                elif e.type == "finished":
                    st["status"] = "succeeded"
                elif e.type == "failed":
                    st["status"] = "failed"
                elif e.type == "info" and "skipped" in e.payload:
                    st["status"] = "skipped"
        return list(steps.values())

    def _verify_outputs(self) -> dict:
        if self.state and self.state.get("plan") is not None:
            for s in self.state["plan"].steps:
                if s.agent == "verifier" and s.outputs:
                    return s.outputs
        return {}

    def detail(self) -> dict[str, Any]:
        st = self.state or {}
        plan = st.get("plan")
        steps = ([{"id": s.id, "agent": s.agent, "objective": s.objective, "status": s.status, "attempts": s.attempts,
                   "depends_on": s.depends_on, "tools": s.tools} for s in plan.steps] if plan is not None else self._steps_from_events())
        ver = self._verify_outputs()
        ex = st.get("execution")
        decision = st.get("decision")
        return {
            "run_id": self.run_id, "request": self.request, "status": self.status, "llm_backend": self.llm_backend,
            "workflow": decision.workflow if decision else None, "created_at": self.created_at, "finished_at": self.finished_at,
            "router": st["router_output"].model_dump(mode="json") if st.get("router_output") else None,
            "decision": decision.model_dump(mode="json") if decision else None, "plan": steps,
            "plan_notes": plan.notes if plan is not None else [], "budget": self.ctx.budget.snapshot(),
            "claims": ver.get("claims", []), "conflicts": ver.get("conflicts", []),
            "failures": ex.failures if ex else [], "recoveries": ex.recoveries if ex else [],
            "sources": [{"source_id": s.source_id, "title": s.title, "url": s.url, "source_type": s.source_type,
                         "published_at": s.published_at.isoformat() if s.published_at else None} for s in self.ctx.sources.all()],
            "partial": st.get("partial_artifact"), "error": self.error or st.get("error"),
            "trace_url": getattr(self.tracer, "trace_url", None), "approval": self.approval, "recall": st.get("recall"),
            "prior_findings": st.get("prior_findings", []), "monitoring_diff": st.get("monitoring_diff"),
        }

    def summary(self) -> dict[str, Any]:
        d = self.detail()
        end = self.finished_at or datetime.now(timezone.utc)
        return {"run_id": self.run_id, "request": self.request, "status": self.status, "workflow": d["workflow"],
                "created_at": self.created_at, "duration_s": (end - self.created_at).total_seconds() if self.done else None,
                "n_sources": len(d["sources"]), "n_supported_claims": sum(c.get("verdict") == "SUPPORTED" for c in d["claims"]),
                "n_conflicts": len(d["conflicts"])}

    def basic_report(self) -> dict[str, Any]:
        """The canonical ResearchReport (as a JSON-able dict) for this run."""
        return build_report(self.request, self.state or {}, self.ctx, status=self.status.value if self.done else None,
                            error=self.error).model_dump(mode="json")


class RunService:
    """Owns the shared (read-only) pipeline components and a worker pool."""

    def __init__(self, *, router, supervisor, agents: dict, llm_factory: Callable[[], Any], tools_factory: Callable[[], Any],
                 runs_dir: str = "runs", llm_backend: str = "unknown", max_workers: int = 4, planner: Planner | None = None,
                 budget_profiles_path: str = "configs/budgets.yaml", memory=None, tracer_factory=make_tracer,
                 approvals: ApprovalPolicy | None = None, record: bool = True):
        self.router, self.supervisor, self.agents = router, supervisor, agents
        self.llm_factory, self.tools_factory = llm_factory, tools_factory
        self.runs_dir, self.llm_backend, self.planner = runs_dir, llm_backend, planner
        self.budget_profiles_path = budget_profiles_path
        self.memory = memory
        self.tracer_factory = tracer_factory
        self.approvals = approvals
        self.record = record
        self.checkpointer = make_checkpointer() if approvals is not None else None
        self._reaper: threading.Thread | None = None
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run")
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def submit(self, text: str, budget_profile: str | None = None) -> RunRecord:
        """Create a run and start it in the background."""
        run_id = uuid.uuid4().hex[:12]
        config: dict = {"llm_backend": self.llm_backend}
        if budget_profile:
            config["budget"] = ExecutionBudget(**load_config(self.budget_profiles_path)["profiles"][budget_profile]).model_dump()
        ctx = RunContext.create(run_id, config, llm=self.llm_factory(), tools=self.tools_factory(), runs_dir=self.runs_dir,
                                memory=self.memory, mode="record" if self.record else "off")
        if self.memory is not None:
            attach_memory(ctx, self.memory, text, self.llm_backend)
        rec = RunRecord(run_id=run_id, request=text, ctx=ctx, llm_backend=self.llm_backend)
        rec.tracer = self.tracer_factory()
        attach_tracing(ctx, rec.tracer, text)
        with self._lock:
            self._runs[run_id] = rec
        self._pool.submit(self._execute, rec)
        return rec

    def _execute(self, rec: RunRecord) -> None:
        rec.status = RunStatus.RUNNING
        rec.ctx.events.emit("run", "started", {"request": rec.request, "llm_backend": rec.llm_backend})
        try:
            rec.graph = build_graph(self.router, self.supervisor, rec.ctx, agents=self.agents, planner=self.planner or Planner(),
                                    recovery=RecoveryManager(), approvals=self.approvals, checkpointer=self.checkpointer)
        except Exception as e:  # noqa: BLE001
            self._crash(rec, e)
            return
        self._drive(rec, initial_state(rec.request))

    def _drive(self, rec: RunRecord, payload) -> None:
        """Run (or resume) the graph. Pauses at an interrupt; otherwise finishes the run."""
        try:
            out = rec.graph.invoke(payload, {"configurable": {"thread_id": rec.run_id}})
            if out.get("__interrupt__"):
                rec.state = {k: v for k, v in out.items() if k != "__interrupt__"}
                self._enter_wait(rec, out["__interrupt__"][0].value)
                return
            rec.state = out
            final = RunStatus(rec.state.get("status", "FAILED"))
            # persist and trace BEFORE publishing the terminal status: a poller that sees `done` must find the stored run
            self._archive(rec, rec.state)
            self._persist(rec, rec.state)
            rec.tracer.end_run(final.value, run_scores(rec.state, rec.ctx))
            rec.finished_at = datetime.now(timezone.utc)
            rec.ctx.budget.freeze()
            rec.status = final
            rec.ctx.events.emit("run", "finished", {"status": final.value})
        except Exception as e:  # noqa: BLE001 - a crashing run must never take the server down
            self._crash(rec, e)

    def _crash(self, rec: RunRecord, e: Exception) -> None:
        log.exception("run %s crashed", rec.run_id)
        rec.error = f"{type(e).__name__}: {e}"
        self._archive(rec, {}, status="FAILED", error=rec.error)
        self._persist(rec, {}, status="FAILED", error=rec.error)       # persist first, then publish the terminal status
        rec.tracer.end_run("FAILED", {})
        rec.finished_at = datetime.now(timezone.utc)
        rec.ctx.budget.freeze()
        rec.status = RunStatus.FAILED
        rec.ctx.events.emit("run", "failed", {"status": "FAILED", "error": rec.error})

    # ---- human-in-the-loop approvals ----
    def _enter_wait(self, rec: RunRecord, request: dict) -> None:
        expires = time.time() + self.approvals.expiry_seconds
        with rec.lock:
            rec.approval = {**request, "requested_at": datetime.now(timezone.utc).isoformat(),
                            "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat()}
            rec.approval_expires = expires
            if self.memory is not None:                                      # persist first, then publish (as for terminal statuses)
                try:
                    self.memory.set_status(rec.run_id, RunStatus.AWAITING_APPROVAL.value)
                except Exception:  # noqa: BLE001
                    log.exception("could not persist AWAITING_APPROVAL for %s", rec.run_id)
            rec.status = RunStatus.AWAITING_APPROVAL
        rec.ctx.events.emit("approval", "approval", {"event": "requested", "kind": request["kind"], "reasons": request.get("reasons", []),
                                                     "searches": request.get("searches"), "expires_at": rec.approval["expires_at"]})
        self._start_reaper()

    def _resolve(self, run_id: str, decision: str, reason: str | None, event: str) -> RunRecord:
        rec = self.get(run_id)
        if rec is None:
            raise KeyError(run_id)
        with rec.lock:
            if rec.status != RunStatus.AWAITING_APPROVAL:
                raise ValueError(f"run {run_id} is not awaiting approval (status {rec.status.value})")
            rec.status, rec.approval, rec.approval_expires = RunStatus.RUNNING, None, None
        rec.ctx.events.emit("approval", "approval", {"event": event, "reason": reason})
        self._pool.submit(self._drive, rec, Command(resume={"decision": decision, "reason": reason}))
        return rec

    def approve(self, run_id: str) -> RunRecord:
        """Resume a run that is waiting for approval. ValueError if it is not waiting; KeyError if unknown."""
        return self._resolve(run_id, "approve", None, "approved")

    def reject(self, run_id: str, reason: str | None = None) -> RunRecord:
        """Cancel a run that is waiting for approval."""
        return self._resolve(run_id, "reject", reason, "rejected")

    def _start_reaper(self) -> None:
        with self._lock:
            if self._reaper is None or not self._reaper.is_alive():
                self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="approval-reaper")
                self._reaper.start()

    def _reap_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for rec in self.list():
                if rec.status == RunStatus.AWAITING_APPROVAL and rec.approval_expires is not None and now > rec.approval_expires:
                    try:
                        self._resolve(rec.run_id, "reject", "approval expired", "expired")
                    except (ValueError, KeyError):
                        pass  # resolved by a person at the same moment
            self._stop.wait(0.1)

    def _archive(self, rec: RunRecord, state: dict, status: str | None = None, error: str | None = None) -> None:
        """Save the recording and run.json (request, config, final report) so the run can be replayed byte-for-byte."""
        if not self.record:
            return
        try:
            rec.ctx.recorder.save()
            report = build_report(rec.request, state, rec.ctx, status=status, error=error).model_dump(mode="json")
            d = Path(self.runs_dir) / rec.run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "run.json").write_text(json.dumps({"run_id": rec.run_id, "request": rec.request, "config": rec.ctx.config,
                                                    "llm_backend": rec.llm_backend, "report": report}, default=str))
        except Exception:  # noqa: BLE001 - archiving must never change a run's outcome
            log.exception("could not archive run %s", rec.run_id)

    def replay(self, run_id: str) -> dict:
        """Re-run a recorded run with tools and LLM forbidden to execute; compare the new report with the original.

        Returns {identical, differences, error, n_calls_replayed}. Raises FileNotFoundError for an unknown / unarchived run.
        """
        from research_os.harness.replay import ReplayMiss, diff_paths, normalize_report

        d = Path(self.runs_dir) / run_id
        meta = json.loads((d / "run.json").read_text())
        ctx = RunContext.create(run_id, meta["config"], llm=self.llm_factory(), tools=self.tools_factory(), runs_dir=None, mode="replay",
                                recording_path=str(d / "recording.jsonl"))
        graph = build_graph(self.router, self.supervisor, ctx, agents=self.agents, planner=self.planner or Planner(), recovery=RecoveryManager())
        error = None
        try:
            state = graph.invoke(initial_state(meta["request"]))
        except ReplayMiss as e:
            state, error = {}, str(e)
        if ctx.recorder.misses and error is None:
            error = ctx.recorder.misses[0]
        report = normalize_report(build_report(meta["request"], state, ctx, status=None if state else "FAILED").model_dump(mode="json"))
        diffs = diff_paths(normalize_report(meta["report"]), report)
        return {"identical": not diffs and error is None, "differences": diffs, "error": error, "n_calls_replayed": ctx.recorder.consumed}

    def _persist(self, rec: RunRecord, state: dict, status: str | None = None, error: str | None = None) -> None:
        if self.memory is None:
            return
        try:
            finalize_run(rec.ctx, self.memory, state, request=rec.request, status=status, error=error)
        except Exception:  # noqa: BLE001 - never let persistence failure change a run's outcome
            log.exception("could not persist run %s", rec.run_id)

    # ---- persisted-only views (runs from before a restart) ----
    def persisted_summary(self, row: dict) -> dict:
        end = row["finished_at"]
        claims = self.memory.get_claims(row["run_id"])
        return {"run_id": row["run_id"], "request": row["request"], "status": row["status"], "workflow": row["workflow"],
                "created_at": row["created_at"], "duration_s": (end - row["created_at"]).total_seconds() if end else None,
                "n_sources": len(self.memory.get_sources(row["run_id"])), "n_supported_claims": sum(c["verdict"] == "SUPPORTED" for c in claims),
                "n_conflicts": len((self.memory.get_report(row["run_id"]) or {}).get("contradictions", []))}

    def persisted_detail(self, run_id: str) -> dict | None:
        if self.memory is None:
            return None
        row = self.memory.get_run(run_id)
        if row is None:
            return None
        report = self.memory.get_report(run_id) or {}
        return {"run_id": run_id, "request": row["request"], "status": row["status"], "llm_backend": row["llm_backend"],
                "workflow": row["workflow"], "created_at": row["created_at"], "finished_at": row["finished_at"], "router": row["router"],
                "decision": None, "plan": [], "plan_notes": [], "budget": row["budget"] or {}, "claims": self.memory.get_claims(run_id),
                "conflicts": report.get("contradictions", []), "failures": [], "recoveries": (report.get("run_metadata") or {}).get("recoveries", []),
                "sources": [{"source_id": s.source_id, "title": s.title, "url": s.url, "source_type": s.source_type,
                             "published_at": s.published_at.isoformat() if s.published_at else None} for s in self.memory.get_sources(run_id)],
                "partial": report if report.get("status") == "PARTIAL" else None,
                "error": report.get("executive_summary") if row["status"] == "FAILED" else None,
                "recall": None, "prior_findings": [], "monitoring_diff": (report.get("run_metadata") or {}).get("monitoring_diff")}

    def get(self, run_id: str) -> RunRecord | None:
        """Run by id."""
        return self._runs.get(run_id)

    def list(self) -> list[RunRecord]:
        """All runs, newest first."""
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def shutdown(self) -> None:
        """Stop accepting work."""
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
