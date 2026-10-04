"""Record / replay of every tool and LLM call, so runs and evaluations are deterministic and offline."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path


class ReplayMiss(RuntimeError):
    """Replay mode was asked for a call that was never recorded."""


class RecordedError(RuntimeError):
    """A non-tool exception captured during recording, re-raised on replay."""


def call_key(kind: str, name: str, args: dict) -> str:
    """Stable key of a call: kind + name + canonical JSON args."""
    return hashlib.sha256(json.dumps([kind, name, args], sort_keys=True, default=str).encode()).hexdigest()


class Recorder:
    """mode: 'off' (pass through), 'record' (run + store), 'replay' (serve stored results, never execute)."""

    def __init__(self, mode: str = "off", path: str | None = None):
        if mode not in ("off", "record", "replay"):
            raise ValueError(f"unknown recorder mode {mode!r}")
        self.mode, self.path = mode, path
        self.entries: list[dict] = []
        self.misses: list[str] = []   # replay requests that were never recorded (the executor may swallow the exception)
        self.consumed = 0
        self._queues: dict[str, deque] = defaultdict(deque)
        if mode == "replay":
            if not path or not Path(path).exists():
                raise FileNotFoundError(f"replay recording not found: {path}")
            for line in Path(path).read_text().splitlines():
                entry = json.loads(line)
                self._queues[entry["key"]].append(entry)

    def call(self, kind: str, name: str, args: dict, fn: Callable[[], dict]) -> dict:
        """Run (or replay) one call. `fn` must return a JSON-able dict."""
        from research_os.contracts import FailureType
        from research_os.tools.registry import ToolError

        if self.mode == "off":
            return fn()
        key = call_key(kind, name, args)
        if self.mode == "replay":
            q = self._queues.get(key)
            if not q:
                msg = f"no recorded {kind} call {name!r} for these arguments (key {key[:12]})"
                self.misses.append(msg)
                raise ReplayMiss(msg)
            entry = q.popleft()
            self.consumed += 1
            if entry["ok"]:
                return entry["result"]
            err = entry["error"]
            if err["type"] == "ToolError":
                raise ToolError(FailureType(err["failure_type"]), err["message"])
            raise RecordedError(f"{err['type']}: {err['message']}")
        entry = {"kind": kind, "name": name, "key": key, "args": args}
        try:
            entry.update(ok=True, result=fn())
            self.entries.append(entry)
            return entry["result"]
        except ToolError as e:
            entry.update(ok=False, error={"type": "ToolError", "failure_type": e.failure_type.value, "message": str(e)})
            self.entries.append(entry)
            raise
        except Exception as e:
            entry.update(ok=False, error={"type": type(e).__name__, "message": str(e)})
            self.entries.append(entry)
            raise

    def save(self) -> None:
        """Write recorded entries as JSONL (record mode only)."""
        if self.mode != "record" or not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(e, default=str) + "\n" for e in self.entries))


# ---------------------------------------------------------------------------------------------------------------------
# `python -m research_os.harness.replay <run_id>`: re-run a recorded run with tools and LLM forbidden to execute, compare reports.
# ---------------------------------------------------------------------------------------------------------------------
def normalize_report(report: dict) -> dict:
    """Copy of a report without wall-clock fields (timestamps and elapsed time), for byte-for-byte comparison."""
    import copy

    r = copy.deepcopy(report)
    meta = r.get("run_metadata") or {}
    meta.pop("generated_at", None)
    (meta.get("budget") or {}).pop("elapsed_seconds", None)
    return r


def diff_paths(a, b, prefix: str = "") -> list[str]:
    """Dotted paths at which two JSON-like values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            out += diff_paths(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in diff_paths(x, y, f"{prefix}[{i}]")]
    return [] if a == b else [prefix or "<root>"]


class _Forbidden:
    """LLM stand-in for replay: any call is a bug (everything must come from the recording)."""

    def complete(self, *a, **k):
        raise RuntimeError("LLM called during replay")


def build_replay_service(runs_dir: str):
    """A RunService whose components can replay recorded runs (no live LLM, no approvals, no memory)."""
    from research_os.agents import build_agents
    from research_os.agents.supervisor import Supervisor
    from research_os.api.service import RunService
    from research_os.router.inference import ResearchRouter
    from research_os.tools import build_registry

    return RunService(router=ResearchRouter.from_config(), supervisor=Supervisor(), agents=build_agents(), llm_factory=_Forbidden,
                      tools_factory=build_registry, runs_dir=runs_dir, llm_backend="replay", record=False)


def replay_main(argv: list[str] | None = None) -> int:
    """CLI: exit 0 if the replayed report is identical (except timestamps), 1 otherwise."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m research_os.harness.replay")
    ap.add_argument("run_id")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)
    result = build_replay_service(args.runs_dir).replay(args.run_id)
    if result["identical"]:
        print(f"IDENTICAL: run {args.run_id} replayed from its recording ({result['n_calls_replayed']} tool/LLM calls, none executed)")
        return 0
    print(f"DIFFERENT: run {args.run_id}")
    if result.get("error"):
        print(f"  error: {result['error']}")
    for d in result["differences"][:20]:
        print(f"  differs at: {d}")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(replay_main(sys.argv[1:]))
