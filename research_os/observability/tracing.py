"""Langfuse tracing as an additive event listener. The local JSONL event log stays the source of truth.

* No-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_HOST for self-hosted; Langfuse Cloud otherwise).
* One trace per run; spans for router, supervisor, planner, every step (agent), tool calls, LLM generations; recovery/budget/retry
  decisions as events; run-level scores. Prompt text is never sent unless `capture_content` is enabled.
* A tracing failure can never fail a run.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from research_os.common import load_config

log = logging.getLogger(__name__)
_SPAN_NODES = ("router", "supervisor", "planner", "recall")
_EVENT_LEVEL = {"recovery": "WARNING", "budget": "WARNING", "retry": "WARNING", "failed": "ERROR"}


class NoopTracer:
    """Tracing disabled: every method does nothing."""

    trace_url: str | None = None

    def start_run(self, run_id: str, request: str) -> None: ...

    def on_event(self, ev) -> None: ...

    def end_run(self, status: str, scores: dict[str, float] | None = None, metadata: dict | None = None) -> None: ...


class LangfuseTracer:
    """Maps RunEvents onto Langfuse observations (root run -> nested spans/agents/tools/generations)."""

    def __init__(self, client, capture_content: bool = False):
        self.client, self.capture_content = client, capture_content
        self._root: Any = None
        self._stack: list[tuple[str, Any]] = []
        self._lock = threading.Lock()
        self.trace_url: str | None = None

    # ---- lifecycle ----
    def start_run(self, run_id: str, request: str) -> None:
        try:
            self._root = self.client.start_observation(name="research_run", as_type="agent", metadata={"run_id": run_id},
                                                       input=request if self.capture_content else None)
        except Exception:  # noqa: BLE001
            log.exception("langfuse: could not start the run trace")
            self._root = None

    def _parent(self):
        return self._stack[-1][1] if self._stack else self._root

    def on_event(self, ev) -> None:
        if self._root is None or ev.node == "run":
            return
        try:
            with self._lock:
                self._handle(ev)
        except Exception:  # noqa: BLE001
            log.exception("langfuse: event handling failed")

    def _handle(self, ev) -> None:
        node, typ, p = ev.node, ev.type, ev.payload
        if typ == "started":
            self._stack.append((node, self._open(node, p)))
        elif typ in ("finished", "failed"):
            idx = next((i for i in range(len(self._stack) - 1, -1, -1) if self._stack[i][0] == node), None)
            if idx is None:
                self._event(ev)
                return
            while len(self._stack) > idx + 1:  # close anything left open below it
                _, child = self._stack.pop()
                child.update(level="WARNING", status_message="ended implicitly")
                child.end()
            _, obs = self._stack.pop()
            if typ == "failed":
                obs.update(level="ERROR", status_message=str(p.get("error", ""))[:500],
                           metadata={"failure_type": p.get("failure_type")})
            elif node.startswith("llm:"):
                obs.update(usage_details={"input": int(p.get("input_tokens", 0)), "output": int(p.get("output_tokens", 0))})
            else:
                obs.update(output={k: v for k, v in p.items() if isinstance(v, (str, int, float, bool))} or None)
            obs.end()
        else:
            self._event(ev)

    def _open(self, node: str, p: dict) -> Any:
        parent = self._parent()
        if node.startswith("tool:"):
            return parent.start_observation(name=node, as_type="tool", metadata={"agent": p.get("agent")},
                                            input=p.get("args") if self.capture_content else None)
        if node.startswith("llm:"):
            meta = {k: v for k, v in p.items() if isinstance(v, (str, int, float, bool))}
            return parent.start_observation(name=node, as_type="generation", model=node[4:], metadata=meta)
        if node.startswith("step:"):
            kind = "evaluator" if p.get("agent") == "verifier" else "agent"
            return parent.start_observation(name=node, as_type=kind, metadata={"agent": p.get("agent"), "attempt": p.get("attempt")})
        return parent.start_observation(name=node, as_type="chain" if node in _SPAN_NODES else "span",
                                        metadata={k: v for k, v in p.items() if isinstance(v, (str, int, float, bool))})

    def _event(self, ev) -> None:
        parent = self._parent()
        action = ev.payload.get("action") if ev.type == "recovery" else None
        name = f"recovery:{action}" if action else f"{ev.type}:{ev.node}"
        parent.create_event(name=name, metadata=ev.payload, level=_EVENT_LEVEL.get(ev.type))

    def end_run(self, status: str, scores: dict[str, float] | None = None, metadata: dict | None = None) -> None:
        if self._root is None:
            return
        try:
            with self._lock:
                while self._stack:
                    _, o = self._stack.pop()
                    o.update(level="WARNING", status_message="ended implicitly")
                    o.end()
            self._root.update(output={"status": status}, metadata=metadata or None, level="ERROR" if status == "FAILED" else None)
            for name, value in (scores or {}).items():
                self._root.score_trace(name=name, value=float(value))
            self._root.end()
            try:
                self.trace_url = self.client.get_trace_url(trace_id=self._root.trace_id)
            except Exception:  # noqa: BLE001
                self.trace_url = None
            self.client.flush()
        except Exception:  # noqa: BLE001
            log.exception("langfuse: could not finish the run trace")


def run_scores(state: dict, ctx) -> dict[str, float]:
    """Run-level scores attached to the trace (verification outcome, effort, success)."""
    plan = state.get("plan")
    ver = next((s.outputs for s in plan.steps if s.agent == "verifier" and s.outputs), {}) if plan is not None else {}
    claims = ver.get("claims", [])
    total = len(claims)
    unsupported = sum(c.get("verdict") == "UNSUPPORTED" for c in claims)
    ex = state.get("execution")
    u = ctx.budget.usage()
    return {"supported_claims": float(sum(c.get("verdict") == "SUPPORTED" for c in claims)),
            "unsupported_claim_rate": unsupported / total if total else 0.0, "conflicts": float(len(ver.get("conflicts", []))),
            "sources": float(len(ctx.sources.all())), "steps_used": float(u["steps"]), "llm_calls": float(u["llm_calls"]),
            "tool_calls": float(u["tool_calls"]), "recoveries": float(len(ex.recoveries) if ex else 0),
            "run_success": 1.0 if state.get("status") == "SUCCESS" else 0.0}


def make_tracer(env: dict | None = None, client_factory=None, cfg_path: str = "configs/tracing.yaml"):
    """LangfuseTracer when both keys are present (host from LANGFUSE_HOST), else a NoopTracer."""
    env = os.environ if env is None else env
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        return NoopTracer()
    cfg = load_config(cfg_path)
    if client_factory is None:
        from langfuse import Langfuse as client_factory  # noqa: N813
    kwargs: dict[str, Any] = {"public_key": env["LANGFUSE_PUBLIC_KEY"], "secret_key": env["LANGFUSE_SECRET_KEY"],
                              "environment": cfg["environment"], "timeout": 5}
    if env.get("LANGFUSE_HOST"):
        kwargs["base_url"] = env["LANGFUSE_HOST"]
    try:
        return LangfuseTracer(client_factory(**kwargs), capture_content=cfg["capture_content"])
    except Exception:  # noqa: BLE001 - bad config must not stop the app
        log.exception("langfuse: could not create the client; tracing disabled")
        return NoopTracer()


def attach_tracing(ctx, tracer, request: str) -> None:
    """Start the run trace and subscribe the tracer to the run's events (no-op tracer adds nothing)."""
    if isinstance(tracer, NoopTracer):
        return
    tracer.start_run(ctx.run_id, request)
    ctx.events.add_listener(tracer.on_event)
