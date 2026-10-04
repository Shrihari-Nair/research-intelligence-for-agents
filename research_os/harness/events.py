"""Run events: in-memory list + JSONL sink (the local source of truth for observability)."""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from research_os.contracts import RunEvent


class EventEmitter:
    """Thread-safe, monotonic-seq event log for one run."""

    def __init__(self, run_id: str, sink_dir: str | None = None):
        self.run_id = run_id
        self.events: list[RunEvent] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[RunEvent], None]] = []
        self._sink: Path | None = None
        if sink_dir:
            d = Path(sink_dir) / run_id
            d.mkdir(parents=True, exist_ok=True)
            self._sink = d / "events.jsonl"

    def add_listener(self, fn: Callable[[RunEvent], None]) -> None:
        """Called synchronously for every new event (used by SSE)."""
        self._listeners.append(fn)

    def emit(self, node: str, type: str, payload: dict | None = None) -> RunEvent:
        """Record one event and return it."""
        with self._lock:
            ev = RunEvent(run_id=self.run_id, seq=len(self.events), ts=datetime.now(timezone.utc), node=node,
                          type=type, payload=payload or {})
            self.events.append(ev)
            if self._sink:
                with self._sink.open("a") as f:
                    f.write(ev.model_dump_json() + "\n")
        for fn in self._listeners:
            fn(ev)
        return ev

    def since(self, seq: int) -> list[RunEvent]:
        """Events with seq >= `seq` (SSE resume)."""
        with self._lock:
            return list(self.events[seq:])
