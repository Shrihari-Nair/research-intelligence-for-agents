"""RunContext: everything an agent needs for one run. Agents never reach for globals."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

from research_os.contracts import Source
from research_os.harness.budgets import BudgetTracker
from research_os.harness.events import EventEmitter
from research_os.harness.replay import Recorder
from research_os.llm.client import LLMResponse
from research_os.tools.registry import ToolRegistry


class SourceStore:
    """Sources gathered in a run, deduplicated by stable source_id."""

    def __init__(self) -> None:
        self._by_id: dict[str, Source] = {}
        self.on_add = None  # optional callback(source) for incremental persistence

    def add(self, source: Source) -> bool:
        """Store a source; False if its id was already present (a longer fetch of the same page replaces a snippet)."""
        old = self._by_id.get(source.source_id)
        if old is not None:
            if len(source.content) > len(old.content):
                self._by_id[source.source_id] = source
                self._notify(source)
            return False
        self._by_id[source.source_id] = source
        self._notify(source)
        return True

    def _notify(self, source: Source) -> None:
        if self.on_add is not None:
            try:
                self.on_add(source)
            except Exception:  # persistence must never break a run
                import logging
                logging.getLogger(__name__).exception("source persistence hook failed")

    def get(self, source_id: str) -> Source | None:
        """Source by id."""
        return self._by_id.get(source_id)

    def all(self) -> list[Source]:
        """All sources in insertion order."""
        return list(self._by_id.values())


class ContextLLM:
    """LLM client bound to a run: counts calls, emits events, records/replays."""

    def __init__(self, base, ctx: RunContext):
        self.base, self.ctx = base, ctx
        self.calls: list[dict] = []  # one entry per completed call: prompt id, model, tokens, latency

    def complete(self, prompt: str, *, model: str | None = None, json_mode: bool = False,
                 max_tokens: int = 1024, meta: dict | None = None) -> LLMResponse:
        ctx = self.ctx
        ctx.budget.count("llm_calls")
        name = model or "default"
        ctx.events.emit(f"llm:{name}", "started", {"prompt_chars": len(prompt), **(meta or {})})
        args = {"prompt": prompt, "model": name, "json_mode": json_mode, "max_tokens": max_tokens}

        def run() -> dict:
            return asdict(self.base.complete(prompt, model=model, json_mode=json_mode, max_tokens=max_tokens, meta=meta))

        try:
            data = ctx.recorder.call("llm", name, args, run)
        except Exception as e:
            ctx.events.emit(f"llm:{name}", "failed", {"error": f"{type(e).__name__}: {e}"})
            raise
        resp = LLMResponse(**data)
        self.calls.append({"prompt": (meta or {}).get("prompt"), "model": resp.model, "input_tokens": resp.input_tokens,
                           "output_tokens": resp.output_tokens, "latency_ms": resp.latency_ms})
        ctx.events.emit(f"llm:{name}", "finished", {"input_tokens": resp.input_tokens,
                                                    "output_tokens": resp.output_tokens})
        return resp


@dataclass
class RunContext:
    """Per-run state passed to every agent and tool."""

    run_id: str
    config: dict
    budget: BudgetTracker
    events: EventEmitter
    recorder: Recorder
    tools: ToolRegistry
    sources: SourceStore = field(default_factory=SourceStore)
    llm: ContextLLM | None = None
    started_at: float = field(default_factory=time.monotonic)
    memory: object | None = None                       # ResearchStore (persistent memory), optional
    step_listeners: list = field(default_factory=list)  # callables(step, outputs) after a step produced outputs

    @classmethod
    def create(cls, run_id: str | None, config: dict, *, llm, tools: ToolRegistry, runs_dir: str | None = None,
               mode: str = "off", recording_path: str | None = None, memory=None) -> RunContext:
        """Build a context. `mode` is record/replay/off; recordings default to <runs_dir>/<run_id>/recording.jsonl."""
        run_id = run_id or uuid.uuid4().hex[:12]
        if recording_path is None and runs_dir and mode != "off":
            recording_path = f"{runs_dir}/{run_id}/recording.jsonl"
        events = EventEmitter(run_id, runs_dir)
        budget = BudgetTracker(on_exhausted=lambda e: events.emit("budget", "budget", {"kind": e.kind, "limit": e.limit, "used": e.used}))
        ctx = cls(run_id=run_id, config=config, budget=budget, events=events, recorder=Recorder(mode, recording_path), tools=tools,
                  memory=memory)
        ctx.llm = ContextLLM(llm, ctx)
        return ctx
