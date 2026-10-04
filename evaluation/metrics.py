"""Per-run scoring and aggregation for the four-system evaluation (quality, agent, efficiency, router metrics)."""
from __future__ import annotations

import numpy as np

from evaluation.validate_dataset import Task
from research_os.common import load_intents
from research_os.graph.templates import load_workflows
from research_os.router.evaluation.metrics import compute_metrics

RESEARCH_KEYS = ("completion", "e2e_success", "unsupported_rate", "citation_correctness", "source_relevance")


def expected_tools(workflow: str) -> set[str]:
    """Tools the workflow template may use (plus fetch_url)."""
    return {t for s in load_workflows()[workflow]["steps"] for t in s.get("tools", [])} | {"fetch_url"}


def completion(run: dict, task: Task) -> dict:
    """Valid complete report: SUCCESS status, required sections, required source types, minimum cited claims."""
    rep = run["report"]
    ev_types = {e["source_type"] for e in run["evidence_checks"]}
    parts = {
        "valid_report": run["status"] == "SUCCESS",
        "sections": bool(rep.get("executive_summary", "").strip()) and len(rep.get("findings", [])) >= 1,
        "source_types": (not task.required_source_types) or bool(ev_types & set(task.required_source_types)),
        "citation_min": len(rep.get("evidence", [])) >= task.min_supported_claims,
    }
    return {**parts, "ok": all(parts.values())}


def score_run(run: dict, task: Task, judged: dict | None, cfg: dict) -> dict:
    """Metrics for one run. `judged` = {"claims": {claim_id: label}, "relevance": {source_id: 0..2}} or None (judge not run)."""
    comp = completion(run, task)
    by = run["tools"]["by_name"]
    total = sum(by.values())
    allowed = expected_tools(task.expected_workflow)
    price = cfg["pricing_per_mtok"]
    llm = run["llm_totals"]
    has_rec = bool(run["recoveries"])
    out = {
        "system": run["system"], "task_id": run["task_id"], "status": run["status"], "kind": task.kind,
        "workflow_correct": run["workflow"] == task.expected_workflow, "completion": comp["ok"], "completion_parts": comp,
        "tool_selection": (sum(n for t, n in by.items() if t in allowed) / total) if total else None,
        "plan_success": run["status"] == "SUCCESS", "has_recovery": has_rec, "recovery_success": (run["status"] == "SUCCESS") if has_rec else None,
        "llm_calls": llm["calls"], "input_tokens": llm["input_tokens"], "output_tokens": llm["output_tokens"],
        "cost_usd": (llm["input_tokens"] * price["input"] + llm["output_tokens"] * price["output"]) / 1e6,
        "tool_calls": run["tools"]["calls"], "searches": run["tools"]["searches"], "steps": run["steps"],
        "routing_latency_ms": (run["routing"]["classifier_ms"] or 0.0) + run["routing_llm_latency_ms"], "wall_ms": run["wall_ms"],
        "router_intent": run["router_intent"], "conflicts": len(run["report"].get("contradictions", [])),
    }
    if judged is not None:
        ev = [e for e in run["evidence_checks"] if e["claim_id"] in judged["claims"]]   # only claims the judge actually labelled
        labels = [judged["claims"].get(e["claim_id"]) for e in ev]
        n = len(ev)
        out["unsupported_rate"] = (sum(lab != "SUPPORTED" for lab in labels) / n) if n else None
        out["citation_correctness"] = (sum(e["verbatim"] and lab == "SUPPORTED" for e, lab in zip(ev, labels, strict=True)) / n) if n else None
        rel = [judged["relevance"][s["source_id"]] for s in run["cited_sources"] if s["source_id"] in judged["relevance"]]
        out["source_relevance"] = (float(np.mean(rel)) / 2) if rel else None
        th = cfg["thresholds"]
        out["e2e_success"] = bool(comp["ok"] and out["citation_correctness"] is not None and out["citation_correctness"] >= th["min_citation_correctness"]
                                  and out["unsupported_rate"] <= th["max_unsupported_rate"])
    return out


def bootstrap_ci(xs: list[float], n: int, confidence: float, seed: int) -> tuple[float | None, float | None]:
    """Seeded percentile bootstrap CI of the mean."""
    if not xs:
        return None, None
    a = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return float(lo), float(hi)


def _stat(xs: list, cfg: dict) -> dict:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "ci95": [None, None]}
    lo, hi = bootstrap_ci(vals, cfg["bootstrap"]["n"], cfg["bootstrap"]["confidence"], cfg["seed"])
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals), "ci95": [lo, hi]}


def _pct(xs: list[float]) -> dict:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "mean": float(a.mean())}


def aggregate(rows: list[dict], cfg: dict, intents_true: list[str], intents_pred: list[str]) -> dict:
    """System-level numbers from per-run scores."""
    intents = load_intents()
    m = compute_metrics(intents_true, intents_pred, intents)
    out: dict = {"n": len(rows), "status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
                 "routing": {"accuracy": float(np.mean([r["workflow_correct"] for r in rows])), "macro_f1": m["macro_f1"],
                             "latency_ms": _pct([r["routing_latency_ms"] for r in rows])},
                 "wall_ms": _pct([r["wall_ms"] for r in rows]), "research": {}, "agent": {}, "efficiency": {}}
    for k in RESEARCH_KEYS:
        if any(k in r for r in rows):
            out["research"][k] = _stat([r.get(k) for r in rows], cfg)
    out["agent"] = {"tool_selection": _stat([r["tool_selection"] for r in rows], cfg), "plan_success": _stat([r["plan_success"] for r in rows], cfg),
                    "recovery_success": _stat([r["recovery_success"] for r in rows], cfg),
                    "runs_with_recovery": sum(r["has_recovery"] for r in rows),
                    "end_to_end_success": out["research"].get("e2e_success") or {"mean": None}}
    out["efficiency"] = {k: _stat([r[k] for r in rows], cfg) for k in ("llm_calls", "input_tokens", "output_tokens", "cost_usd", "tool_calls", "searches", "steps")}
    out["efficiency"]["total_cost_usd"] = float(sum(r["cost_usd"] for r in rows))
    return out
