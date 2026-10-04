"""Record the frozen evaluation corpus: run every task ONCE live (real tools + real LLM) and store its cassette and source pool.

Usage: python -m evaluation.record_corpus [--ids t001 t002] [--limit N] [--force] [--workers 3]
Writes per task: evaluation/cassettes/<id>.jsonl (tool+LLM call recording), <id>.sources.json (every source seen), <id>.meta.json.
Also fills evaluation/cassettes/llm_cache.jsonl so later replays are identical and free.
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from evaluation.replay import CachedLLM
from evaluation.systems import build_pipeline, run_pipeline
from evaluation.validate_dataset import CASSETTES, Task, load_tasks
from research_os.common import get_hardware_info, load_dotenv, setup_logging
from research_os.harness.context import RunContext
from research_os.llm.gemini_client import GeminiLLMClient
from research_os.llm.models import model_for_role, provider_settings
from research_os.router.inference import ResearchRouter
from research_os.tools import build_registry

log = logging.getLogger(__name__)


def record_task(task: Task, router, llm, registry, out_dir: Path) -> dict:
    """Run one task live with recording; always leaves a cassette + sources file (empty on total failure)."""
    cas = out_dir / f"{task.id}.jsonl"
    ctx = RunContext.create(task.id, {"llm_backend": "gemini"}, llm=llm, tools=registry, mode="record", recording_path=str(cas))
    meta: dict = {"task_id": task.id, "recorded_at": datetime.now(timezone.utc).isoformat()}
    try:
        state = run_pipeline(build_pipeline("D", router), task.input, ctx)
        meta.update(status=state.get("status"), workflow=state["decision"].workflow if state.get("decision") else None,
                    router_intent=state["router_output"].intent if state.get("router_output") else None,
                    failures=[(f["step"], f["failure_type"]) for f in state["execution"].failures] if state.get("execution") else [],
                    recoveries=[(r["step"], r["action"]) for r in state["execution"].recoveries] if state.get("execution") else [],
                    halted_reason=state["execution"].halted_reason if state.get("execution") else None)
    except Exception as e:  # noqa: BLE001 - a crashing task still gets a (possibly empty) cassette
        log.exception("task %s crashed while recording", task.id)
        meta.update(status="CRASHED", error=f"{type(e).__name__}: {e}")
    ctx.recorder.save()
    cas.touch()
    sources = [s.model_dump(mode="json") for s in ctx.sources.all()]
    (out_dir / f"{task.id}.sources.json").write_text(json.dumps(sources))
    meta.update(n_sources=len(sources), budget=ctx.budget.snapshot())
    (out_dir / f"{task.id}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    setup_logging()
    load_dotenv()
    out = Path(CASSETTES)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in load_tasks() if not args.ids or t.id in args.ids]
    if not args.force:
        tasks = [t for t in tasks if not (out / f"{t.id}.sources.json").exists()]
    tasks = tasks[: args.limit] if args.limit else tasks
    prov = provider_settings()
    llm = CachedLLM(GeminiLLMClient(model_for_role("supervisor"), prov["api_key_env"], prov["timeout_s"] * 2), str(out / "llm_cache.jsonl"))
    router = ResearchRouter.from_config()
    registry = build_registry(mode="live")
    log.info("recording %d task(s) with %d worker(s)", len(tasks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for meta in pool.map(lambda t: record_task(t, router, llm, registry, out), tasks):
            log.info("%s %s sources=%s", meta["task_id"], meta.get("status"), meta.get("n_sources"))
    (out / "RECORDING.json").write_text(json.dumps({"recorded_with": "live tools (Tavily, GitHub, arXiv, PubMed, fetch_url) + Gemini",
                                                    "date": datetime.now(timezone.utc).date().isoformat(), "hardware": get_hardware_info(),
                                                    "note": "web content drifts; evaluation replays these frozen sources"}, indent=1))


if __name__ == "__main__":
    main()
