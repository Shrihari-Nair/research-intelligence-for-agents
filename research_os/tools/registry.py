"""ToolRegistry: the only way agents call tools. Enforces schemas, permissions, policies, timeouts, retries, output caps.

Every call is counted against the run budget, recorded for replay, and emits events.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError
from pydantic import Field as PField

from research_os.contracts import FailureType, Source


class ToolError(Exception):
    """A classified tool failure (recovery keys on `failure_type`)."""

    def __init__(self, failure_type: FailureType, message: str):
        super().__init__(message)
        self.failure_type = failure_type


class ToolResult(BaseModel):
    """What every tool returns: sources and/or structured data."""

    sources: list[Source] = PField(default_factory=list)
    data: dict = PField(default_factory=dict)


@dataclass
class ToolSpec:
    """A registered tool and its enforcement settings."""

    name: str
    fn: Callable[[dict, object], ToolResult]
    cost_class: str = "other"            # 'search' counts against max_searches
    input_schema: type[BaseModel] | None = None
    timeout: float | None = None         # seconds; None -> registry default
    max_retries: int | None = None       # None -> registry default
    permissions: tuple[str, ...] | list[str] = ("research",)
    max_content_chars: int | None = None
    pre_checks: tuple[Callable[[dict], None], ...] = field(default_factory=tuple)


class ToolRegistry:
    """Dispatches tool calls with enforcement. `permissions=None` disables permission checks (unit tests of tools)."""

    def __init__(self, retry_cfg: dict | None = None, permissions: dict[str, list[str]] | None = None,
                 default_timeout: float = 15.0):
        self._tools: dict[str, ToolSpec] = {}
        r = retry_cfg or {}
        self.max_retries = int(r.get("max_retries", 0))
        self.backoff_s = float(r.get("backoff_s", 0.0))
        self.retry_on = set(r.get("retry_on", []))
        self.permissions = permissions
        self.default_timeout = default_timeout

    def register(self, name: str, fn: Callable[[dict, object], ToolResult], cost_class: str = "other") -> None:
        """Convenience for simple tools (no schema, default limits)."""
        self.register_spec(ToolSpec(name=name, fn=fn, cost_class=cost_class))

    def register_spec(self, spec: ToolSpec) -> None:
        """Register (or replace) a tool."""
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        """Registered tool names."""
        return sorted(self._tools)

    # ------------------------------------------------------------------
    def _execute(self, spec: ToolSpec, args: dict, ctx) -> ToolResult:
        timeout = spec.timeout if spec.timeout is not None else self.default_timeout
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(spec.fn, args, ctx)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout as e:
            raise ToolError(FailureType.SEARCH_TIMEOUT, f"{spec.name} exceeded its {timeout}s timeout") from e
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(FailureType.TOOL_FAILURE, f"{type(e).__name__}: {e}") from e
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _cost(spec: ToolSpec) -> dict[str, int]:
        return {"tool_calls": 1, "searches": 1} if spec.cost_class == "search" else {"tool_calls": 1}

    def call(self, name: str, args: dict, ctx, agent: str | None = None) -> ToolResult:
        """Dispatch one call. Raises ToolError (never any other exception) on failure."""
        spec, node = self._tools.get(name), f"tool:{name}"

        def fail(ft: FailureType, msg: str):
            ctx.events.emit(node, "failed", {"failure_type": ft.value, "error": msg})
            raise ToolError(ft, msg)

        if spec is None:
            fail(FailureType.TOOL_FAILURE, f"unknown tool {name!r}")
        if agent is not None and self.permissions is not None:
            granted = set(self.permissions.get(agent, []))
            if not (set(spec.permissions) & granted):
                fail(FailureType.TOOL_FAILURE, f"permission denied: agent {agent!r} may not call {name!r}")
        if spec.input_schema is not None:
            try:
                args = spec.input_schema.model_validate(args).model_dump()
            except ValidationError as e:
                fail(FailureType.TOOL_FAILURE, f"invalid arguments for {name}: {str(e).splitlines()[0]} ({e.error_count()} error(s))")
        for check in spec.pre_checks:
            try:
                check(args)
            except ToolError as e:
                fail(e.failure_type, str(e))
        retries = spec.max_retries if spec.max_retries is not None else self.max_retries
        ctx.budget.count_all(self._cost(spec))
        ctx.events.emit(node, "started", {"args": args, "agent": agent})

        def run() -> dict:
            last: ToolError | None = None
            for attempt in range(retries + 1):
                if attempt:
                    ctx.budget.count_all(self._cost(spec))
                    ctx.events.emit(node, "retry", {"attempt": attempt, "error": str(last)})
                    time.sleep(self.backoff_s * 2 ** (attempt - 1))
                try:
                    res = self._execute(spec, args, ctx)
                    cap = spec.max_content_chars
                    if cap:
                        res = res.model_copy(update={"sources": [s.model_copy(update={"content": s.content[:cap]}) for s in res.sources]})
                    return res.model_dump(mode="json")
                except ToolError as e:
                    last = e
                    if e.failure_type.value not in self.retry_on or attempt == retries:
                        raise
            raise last  # pragma: no cover - loop always returns or raises

        try:
            data = ctx.recorder.call("tool", name, args, run)
        except ToolError as e:
            ctx.events.emit(node, "failed", {"failure_type": e.failure_type.value, "error": str(e)})
            raise
        result = ToolResult.model_validate(data)
        for s in result.sources:
            ctx.sources.add(s)
        ctx.events.emit(node, "finished", {"sources": len(result.sources)})
        return result
