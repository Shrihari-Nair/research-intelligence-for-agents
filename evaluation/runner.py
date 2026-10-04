"""Run every task on every system. Default is REPLAY: frozen per-task sources + a content-addressed LLM cache (cache-then-live).

Usage: python -m evaluation.runner [--systems A B C D ...] [--ablations] [--limit N] [--ids t001 ...] [--workers 3] [--live] [--offline] [--force]
  --live     real tools and a fresh (uncached) LLM: for a small subset only
  --offline  never call the LLM; a cache miss is an error (proves a rerun is fully reproducible)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.replay import CachedLLM, CassetteCorpus, build_replay_registry
from evaluation.systems import SYSTEMS, build_pipeline, run_pipeline
from evaluation.validate_dataset import Task, load_tasks
from research_os.agents.text import find_quote
from research_os.artifacts.report import build_report
from research_os.common import get_hardware_info, load_config, load_dotenv, set_seed, setup_logging, write_json
from research_os.harness.context import RunContext
from research_os.tools import build_registry

log = logging.getLogger(__name__)
# Everything below is identical across replays of the same cache; wall-clock fields (wall_ms, classifier_ms) are not.
DETERMINISTIC_KEYS = ["system", "task_id", "status", "workflow", "router_intent", "routing_mode", "supervisor_llm_used", "steps",
                      "recoveries", "failures", "halted_reason", "report_digest", "tools", "evidence_checks", "llm_totals",
                      "routing_llm_calls", "routing_llm_latency_ms", "cited_sources"]


def _digest(report: dict) -> str:
    keep = {k: report.get(k) for k in ("executive_summary", "findings", "evidence", "contradictions", "sources", "comparison")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def run_path(cfg: dict, system: str, task_id: str) -> Path:
    return Path(cfg["paths"]["runs"]) / system / f"{task_id}.json"


def load_run(cfg: dict, system: str, task_id: str) -> dict:
    """A saved run record."""
    return json.loads(run_path(cfg, system, task_id).read_text())


def run_one(system: str, task: Task, cfg: dict, router, base_llm, *, llm: CachedLLM | None = None, live: bool = False) -> dict:
    """Run one task on one system and return its record (never raises)."""
    llm_client = llm or (base_llm if live else CachedLLM(base_llm, cfg["paths"]["llm_cache"]))
    if live:
        registry = build_registry(mode="live")
    else:
        registry = build_replay_registry(CassetteCorpus.load(task.id, cfg["paths"]["cassettes"]))
    ctx = RunContext.create(f"{system}-{task.id}", {"llm_backend": "gemini"}, llm=llm_client, tools=registry)
    rec: dict = {"system": system, "task_id": task.id, "mode": "live" if live else "replay"}
    t0 = time.perf_counter()
    try:
        state = run_pipeline(build_pipeline(system, router), task.input, ctx)
    except Exception as e:  # noqa: BLE001 - a crash is a data point, not a reason to stop the evaluation
        log.exception("%s/%s crashed", system, task.id)
        state = {}
        rec.update(error=f"{type(e).__name__}: {e}")
    wall_ms = (time.perf_counter() - t0) * 1000
    report = build_report(task.input, state, ctx, llm_backend="gemini", status=None if state else "FAILED",
                          error=rec.get("error")).model_dump(mode="json")
    ro, dec, ex = state.get("router_output"), state.get("decision"), state.get("execution")
    calls = ctx.llm.calls
    sup = [c for c in calls if c["prompt"] == "supervisor.v1"]
    tool_names: dict[str, int] = {}
    for e in ctx.events.events:
        if e.node.startswith("tool:") and e.type == "started":
            tool_names[e.node[5:]] = tool_names.get(e.node[5:], 0) + 1
    usage = ctx.budget.usage()
    totals = {"calls": len(calls), "input_tokens": sum(c["input_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls),
              "latency_ms": round(sum(c["latency_ms"] for c in calls), 3),
              "by_prompt": {p: sum(c["prompt"] == p for c in calls) for p in sorted({str(c["prompt"]) for c in calls})}}
    checks = []
    for ev in report["evidence"]:
        s = ctx.sources.get(ev["source_id"])
        checks.append({"claim_id": ev["claim_id"], "claim": ev["claim"], "quote": ev["evidence_quote"], "source_id": ev["source_id"],
                       "source_title": s.title if s else "", "source_type": s.source_type if s else "",
                       "verbatim": bool(s and find_quote(ev["evidence_quote"], s.content))})
    cited, seen = [], set()
    for ev in report["evidence"]:
        s = ctx.sources.get(ev["source_id"])
        if s and s.source_id not in seen:
            seen.add(s.source_id)
            cited.append({"source_id": s.source_id, "title": s.title, "source_type": s.source_type, "snippet": s.content[:600]})
    rec.update({
        "cited_sources": cited, "status": report["status"], "workflow": dec.workflow if dec else None, "router_intent": ro.intent if ro else None,
        "routing_mode": ro.routing_mode.value if ro else None, "supervisor_llm_used": bool(dec.llm_used) if dec else False,
        "steps": usage["steps"], "recoveries": [{"step": r["step"], "action": r["action"]} for r in (ex.recoveries if ex else [])],
        "failures": [{"step": f["step"], "failure_type": f["failure_type"]} for f in (ex.failures if ex else [])],
        "halted_reason": ex.halted_reason if ex else None, "report": report, "report_digest": _digest(report),
        "tools": {"calls": usage["tool_calls"], "searches": usage["searches"], "by_name": dict(sorted(tool_names.items()))},
        "llm_totals": totals, "llm": {**totals, "cache": llm_client.stats() if hasattr(llm_client, "stats") else None},
        "routing": {"classifier_ms": ro.latency_ms if ro else None, "llm_calls": len(sup), "llm_latency_ms": round(sum(c["latency_ms"] for c in sup), 3)},
        "routing_llm_calls": len(sup), "routing_llm_latency_ms": round(sum(c["latency_ms"] for c in sup), 3),
        "evidence_checks": checks, "wall_ms": wall_ms})
    return rec


def run_tasks(systems: list[str], tasks: list[Task], cfg: dict, router, base_llm, *, workers: int = 3, force: bool = False,
              live: bool = False) -> list[dict]:
    """Run every (system, task) pair; saved records are reused unless `force`. Returns all records."""
    shared = None if live else CachedLLM(base_llm, cfg["paths"]["llm_cache"])
    pairs = [(s, t) for s in systems for t in tasks]
    out: list[dict] = []

    def work(pair):
        s, t = pair
        p = run_path(cfg, s, t.id)
        if p.exists() and not force:
            return json.loads(p.read_text())
        rec = run_one(s, t, cfg, router, base_llm, llm=shared, live=live)
        write_json(p, rec)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(work, pairs):
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*")
    ap.add_argument("--ablations", action="store_true", help="also run the D ablations")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    setup_logging()
    load_dotenv()
    cfg = load_config(args.config)
    set_seed(args.seed if args.seed is not None else cfg["seed"])
    systems = args.systems or (cfg["systems"] + (cfg["ablations"] if args.ablations else []))
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}")
    tasks = [t for t in load_tasks(cfg["paths"]["tasks"]) if not args.ids or t.id in args.ids]
    tasks = tasks[: args.limit] if args.limit else tasks
    base = None
    if not args.offline:
        from research_os.llm.gemini_client import GeminiLLMClient
        from research_os.llm.models import model_for_role, provider_settings
        prov = provider_settings()
        base = GeminiLLMClient(model_for_role("supervisor"), prov["api_key_env"], prov["timeout_s"] * 2)
    from research_os.router.inference import ResearchRouter
    router = ResearchRouter.from_config()
    log.info("running %d system(s) x %d task(s), mode=%s", len(systems), len(tasks), "live" if args.live else "replay")
    recs = run_tasks(systems, tasks, cfg, router, base, workers=args.workers, force=args.force, live=args.live)
    write_json(f"{cfg['paths']['results']}/run_manifest.json", {"systems": systems, "n_tasks": len(tasks), "n_runs": len(recs),
               "mode": "live" if args.live else "replay", "seed": cfg["seed"], "hardware": get_hardware_info()})
    log.info("done: %d runs", len(recs))


if __name__ == "__main__":
    main()
