"""Execution budgets: counters + hard limits checked BEFORE every step, tool call and LLM call (hard rule 6)."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from research_os.contracts import ExecutionBudget, FailureType

_LIMIT_FIELD = {"steps": "max_steps", "tool_calls": "max_tool_calls", "searches": "max_searches", "llm_calls": "max_llm_calls"}


class BudgetExhausted(Exception):
    """A budget limit was reached. Converted to a BUDGET_EXHAUSTED failure and handed to recovery, never an unhandled crash."""

    failure_type = FailureType.BUDGET_EXHAUSTED

    def __init__(self, kind: str, limit: float, used: float):
        super().__init__(f"budget exhausted: {kind} limit {limit} reached (used {used})")
        self.kind, self.limit, self.used = kind, limit, used


class BudgetTracker:
    """Counts steps, tool calls, searches and LLM calls and wall-clock time; enforces limits once they are set."""

    COUNTERS = ("steps", "tool_calls", "searches", "llm_calls")

    def __init__(self, budget: ExecutionBudget | None = None, clock: Callable[[], float] = time.monotonic,
                 on_exhausted: Callable[[BudgetExhausted], None] | None = None) -> None:
        self._n = dict.fromkeys(self.COUNTERS, 0)
        self._clock, self._start = clock, clock()
        self._frozen: float | None = None
        self._lock = threading.Lock()
        self._budget = budget
        self.on_exhausted = on_exhausted

    def set_limits(self, budget: ExecutionBudget | None) -> None:
        """Apply (or replace) the limits; the clock keeps running from run start."""
        self._budget = budget

    def limits(self) -> ExecutionBudget | None:
        """Current limits (None while unlimited)."""
        return self._budget

    def _refuse(self, exc: BudgetExhausted) -> None:
        if self.on_exhausted:
            self.on_exhausted(exc)
        raise exc

    def count(self, kind: str, n: int = 1) -> None:
        """Check the limits, then increment `kind`. Raises BudgetExhausted (without incrementing) when over a limit."""
        self.count_all({kind: n})

    def count_all(self, kinds: dict[str, int]) -> None:
        """Atomically check every limit, then increment all `kinds`; a refusal increments nothing."""
        for kind in kinds:
            if kind not in self._n:
                raise KeyError(f"unknown budget counter {kind!r}")
        exc: BudgetExhausted | None = None
        with self._lock:
            b = self._budget
            if b is not None:
                elapsed = self._clock() - self._start
                if elapsed >= b.max_runtime_seconds:
                    exc = BudgetExhausted("runtime", b.max_runtime_seconds, round(elapsed, 3))
                else:
                    for kind, n in kinds.items():
                        limit = getattr(b, _LIMIT_FIELD[kind])
                        if self._n[kind] + n > limit:
                            exc = BudgetExhausted(kind, limit, self._n[kind])
                            break
            if exc is None:
                for kind, n in kinds.items():
                    self._n[kind] += n
        if exc is not None:
            self._refuse(exc)

    def freeze(self) -> None:
        """Stop the elapsed clock (the run finished); later reads keep reporting the run's own duration."""
        with self._lock:
            if self._frozen is None:
                self._frozen = self._clock() - self._start

    def usage(self) -> dict:
        """Current counters plus elapsed seconds."""
        with self._lock:
            return {**self._n, "elapsed_seconds": self._frozen if self._frozen is not None else self._clock() - self._start}

    def snapshot(self) -> dict:
        """Usage plus limits, for events and report metadata."""
        return {**self.usage(), "limits": self._budget.model_dump() if self._budget else None}
