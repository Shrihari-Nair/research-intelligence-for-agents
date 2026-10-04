"""Compare LLM-only (A), MiniLM-only (B) and Hybrid (C) on test, hard_test and e2e scenarios."""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

from research_os.common import get_hardware_info, intents_from_cfg, load_config, setup_logging, write_json
from research_os.legacy.agents.tools import ToolBox
from research_os.legacy.factory import build_llm, build_system
from research_os.legacy.graph.workflow import build_graph, run_text
from research_os.router.data.load import load_split
from research_os.router.evaluation.e2e_tasks import E2E_TASKS, task_success
from research_os.router.evaluation.metrics import compute_metrics
from research_os.router.loader import load_router

log = logging.getLogger(__name__)
NOT_RUN = "not_run"


def cost_per_request(records: list[dict], pricing: dict) -> float:
    """Mean USD per request from actual token usage and configured per-million-token prices."""
    total = sum(r["input_tokens"] * pricing["input"] + r["output_tokens"] * pricing["output"] for r in records)
    return total / 1e6 / max(len(records), 1)


def run_routing_eval(system, df, intents: list[str], pricing: dict) -> dict:
    """Route every row of `df`; return intent metrics, latency, CPU, LLM calls and cost."""
    recs, cpu0, wall0 = [], time.process_time(), time.perf_counter()
    for text in df["text"]:
        recs.append(system.route(text))
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    m = compute_metrics(df["intent"].tolist(), [r["intent"] for r in recs], intents)
    lat = np.array([r["latency_ms"] for r in recs])
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "emergency_recall": m["emergency_recall"],
            "per_intent_f1": m["per_intent_f1"], "confusion_matrix": m["confusion_matrix"],
            "input_tokens": sum(r["input_tokens"] for r in recs),
            "output_tokens": sum(r["output_tokens"] for r in recs),
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "cpu_ms_per_request": cpu / len(recs) * 1000, "cpu_utilization_pct": cpu / wall * 100,
            "cost_per_request": cost_per_request(recs, pricing),
            "llm_calls": sum(r["llm_called"] for r in recs), "llm_failures": sum(r["llm_failed"] for r in recs),
            "n": len(recs)}


def run_e2e(system, tasks: list[dict], llm, toolbox_factory=ToolBox) -> dict:
    """Run each task through the full graph with a fresh toolbox; success = agent + tool correct."""
    ok, lat = 0, []
    for t in tasks:
        state = run_text(build_graph(system, toolbox_factory(), llm), t["text"])
        ok += task_success(state, t)
        lat.append(state["routing"]["latency_ms"])
    return {"task_success": ok / len(tasks), "n": len(tasks), "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95))}


def _fmt(v, kind="num") -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "n/a"
    return {"pct": f"{v * 100:.1f}%", "ms": f"{v:.1f} ms", "usd": f"${v:.6f}", "num": f"{v:g}"}[kind]


def render_markdown(res: dict) -> str:
    """One table per dataset in the roadmap's layout."""
    systems = ["llm_only", "minilm_only", "hybrid"]
    names = {"llm_only": "LLM-only", "minilm_only": "MiniLM-only", "hybrid": "Hybrid"}
    rows = [("Intent accuracy", "accuracy", "pct"), ("Macro F1", "macro_f1", "pct"),
            ("EMERGENCY recall", "emergency_recall", "pct"), ("P50 latency", "p50_ms", "ms"),
            ("P95 latency", "p95_ms", "ms"), ("CPU time / request", "cpu_ms_per_request", "ms"),
            ("Cost / request", "cost_per_request", "usd"), ("LLM calls", "llm_calls", "num")]
    out = [f"LLM backend used for hybrid fallback: **{res['llm_backend']}**.", ""]
    for ds in ("test", "hard_test"):
        out += [f"### {ds}", "", "| Metric | " + " | ".join(names[s] for s in systems) + " |",
                "|---|---|---|---|"]
        for label, key, kind in rows:
            out.append(f"| {label} | " + " | ".join(_fmt(res[ds][s].get(key, NOT_RUN) if isinstance(res[ds][s], dict)
                                                       else res[ds][s], kind) for s in systems) + " |")
        out.append("")
    out += ["### End-to-end task success", "", "| | " + " | ".join(names[s] for s in systems) + " |",
            "|---|---|---|---|", "| Task success | " + " | ".join(
                _fmt(res["e2e"][s]["task_success"], "pct") if isinstance(res["e2e"][s], dict) else NOT_RUN
                for s in systems) + " |", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--routing", default="configs/routing.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg, rcfg = load_config(args.config), load_config(args.routing)
    intents = intents_from_cfg(cfg)
    torch.set_num_threads(cfg["benchmark"]["threads"])
    log.info("torch threads=%d", torch.get_num_threads())
    router, _, _ = load_router(args.model or cfg["paths"]["model"], cfg["paths"]["baseline_model"])
    llm, backend = build_llm(rcfg)
    pricing = rcfg["pricing_per_mtok"]
    systems = {"minilm_only": build_system("minilm", router, llm, rcfg),
               "hybrid": build_system("hybrid", router, llm, rcfg)}
    if backend == "gemini":
        systems["llm_only"] = build_system("llm", router, llm, rcfg)
    else:  # never simulate LLM numbers
        log.warning("no Gemini key: LLM-only marked %s; hybrid uses the mock fallback", NOT_RUN)

    res: dict = {"llm_backend": backend, "threshold": rcfg["threshold"], "emergency_floor": rcfg["emergency_floor"],
                 "pricing_per_mtok_note": "configurable, not authoritative", "pricing_per_mtok": pricing,
                 "hardware": get_hardware_info()}
    for ds in ("test", "hard_test"):
        df = load_split(ds, cfg["paths"]["processed"])
        res[ds] = {"llm_only": NOT_RUN}
        for name, sysm in systems.items():
            res[ds][name] = run_routing_eval(sysm, df, intents, pricing)
            if name == "hybrid" and backend == "mock":  # mock tokens are fake, so cost is not meaningful
                res[ds][name]["cost_per_request"] = "mock_backend"
            log.info("%s/%s acc=%.4f", ds, name, res[ds][name]["accuracy"])
    res["e2e"] = {"llm_only": NOT_RUN}
    for name, sysm in systems.items():
        res["e2e"][name] = run_e2e(sysm, E2E_TASKS, llm)
        log.info("e2e/%s success=%.3f", name, res["e2e"][name]["task_success"])
    out = cfg["paths"]["results"]
    write_json(f"{out}/compare_systems.json", res)
    with open(f"{out}/compare_systems.md", "w") as f:
        f.write(render_markdown(res))
    print(render_markdown(res))


if __name__ == "__main__":
    main()
