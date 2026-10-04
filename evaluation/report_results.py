"""Judge the runs, score and aggregate every system, render experiments/results/summary.{json,md}.

Usage: python -m evaluation.report_results [--offline]     (offline: every judge call must already be cached)
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from evaluation.judges import Judge, agreement
from evaluation.metrics import RESEARCH_KEYS, aggregate, score_run
from evaluation.replay import CachedLLM
from evaluation.validate_dataset import load_tasks
from research_os.common import get_hardware_info, load_config, load_dotenv, setup_logging, write_json
from research_os.llm.models import model_for_role

log = logging.getLogger(__name__)
NOT_RUN = "not_run"


def _load_runs(cfg: dict, system: str) -> list[dict]:
    d = Path(cfg["paths"]["runs"]) / system
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def judge_runs(runs: list[dict], cfg: dict, base_llm, workers: int = 4) -> dict[tuple[str, str], dict]:
    """LLM-judge every run's evidence and cited sources. Results are cached, so reruns are free and identical."""
    llm = CachedLLM(base_llm, cfg["paths"]["judge_cache"])
    judge = Judge(llm, batch_size=cfg["judge"]["batch_size"])

    def one(r: dict):
        ev = r["evidence_checks"][: cfg["judge"]["max_evidence_per_run"]]
        claims = judge.claim_support([{"id": e["claim_id"], "claim": e["claim"], "quote": e["quote"], "source": e["source_title"]} for e in ev])
        src = r["cited_sources"][: cfg["judge"]["max_sources_per_run"]]
        rel = judge.source_relevance(r["report"]["question"], [{"id": s["source_id"], "title": s["title"], "snippet": s["snippet"]} for s in src]) if src else {}
        return (r["system"], r["task_id"]), {"claims": claims, "relevance": rel}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, runs))


def _flat(agg: dict) -> dict[str, float | None]:
    out = {}
    for sec in ("research", "agent", "efficiency"):
        for k, v in agg[sec].items():
            if isinstance(v, dict) and "mean" in v:
                out[f"{sec}.{k}"] = v["mean"]
    out["routing.accuracy"] = agg["routing"]["accuracy"]
    return out


def build_summary(cfg: dict, base_llm, use_judge: bool = True) -> dict:
    """Everything the README and docs report, computed from saved runs."""
    tasks = {t.id: t for t in load_tasks(cfg["paths"]["tasks"])}
    names = [*cfg["systems"], *cfg["ablations"]]
    runs_by = {n: [r for r in _load_runs(cfg, n) if r["task_id"] in tasks] for n in names}
    all_runs = [r for rs in runs_by.values() for r in rs]
    judged = judge_runs(all_runs, cfg, base_llm) if (use_judge and all_runs) else None
    scores: dict[str, list[dict]] = {}
    agg: dict[str, dict | str] = {}
    for n, rs in runs_by.items():
        if not rs:
            agg[n] = NOT_RUN
            continue
        rows = [score_run(r, tasks[r["task_id"]], judged.get((n, r["task_id"])) if judged else None, cfg) for r in rs]
        scores[n] = rows
        agg[n] = aggregate(rows, cfg, [tasks[r["task_id"]].expected_intent for r in rs], [(r["workflow"] or "general_research").upper() for r in rs])
        agg[n]["by_kind"] = {k: {m: (float(sum(x[m] for x in rows if x["kind"] == k and x.get(m) is not None) / max(1, sum(
            x["kind"] == k and x.get(m) is not None for x in rows)))) for m in RESEARCH_KEYS if any(m in x for x in rows)}
            for k in ("normal", "ambiguous", "tempting")}
    ablations: dict[str, dict | str] = {}
    for n in cfg["ablations"]:
        if agg[n] == NOT_RUN or agg.get("D") == NOT_RUN or "D" not in agg:
            ablations[n] = NOT_RUN
            continue
        a, d = _flat(agg[n]), _flat(agg["D"])
        ablations[n] = {"delta_vs_D": {"research": {k.split(".", 1)[1]: (a[k] - d[k]) for k in a if k.startswith("research.") and a[k] is not None and d.get(k) is not None},
                                       "metrics": {k: (a[k] - d[k]) for k in a if a[k] is not None and d.get(k) is not None}}}
    gold_path = Path(cfg["paths"]["gold_claims"])
    ag: dict | str = NOT_RUN
    if judged and gold_path.exists() and ("D", next(iter(tasks))) is not None:
        gold = {f"{g['task_id']}|{g['claim_id']}": g["label"] for g in map(json.loads, filter(None, gold_path.read_text().splitlines()))}
        lab = {f"{t}|{c}": v for (s, t), j in judged.items() if s == "D" for c, v in j["claims"].items()}
        ag = agreement(lab, gold)
    summary = {"generated": date.today().isoformat(), "n_tasks": len(tasks), "systems": {n: agg[n] for n in cfg["systems"]}, "ablations": ablations,
               "ablation_systems": {n: agg[n] for n in cfg["ablations"]}, "judge": {"model": model_for_role("eval_judge"), "agreement": ag if judged else NOT_RUN},
               "config": {"thresholds": cfg["thresholds"], "pricing_per_mtok": cfg["pricing_per_mtok"], "seed": cfg["seed"]},
               "hardware": get_hardware_info()}
    write_json(f"{cfg['paths']['results']}/scores.json", {n: rows for n, rows in scores.items()})
    return summary


def _f(stat, pct=False) -> str:
    if not isinstance(stat, dict) or stat.get("mean") is None:
        return "n/a"
    m, (lo, hi) = stat["mean"], stat["ci95"]
    v = (lambda x: f"{x * 100:.1f}%") if pct else (lambda x: f"{x:.2f}")
    return f"{v(m)} [{v(lo)}, {v(hi)}]" if lo is not None else v(m)


def _row(name, cells) -> str:
    return f"| {name} | " + " | ".join(cells) + " |"


def render_summary_md(s: dict) -> str:
    """Markdown tables for README/docs. Every number comes from the saved runs."""
    sys = s["systems"]
    names = list(sys)
    ag = s["judge"]["agreement"]
    ag_txt = (f"n={ag['n']}, accuracy={ag['accuracy']:.2f}" + (f", kappa={ag['kappa']:.2f}" if ag.get("kappa") is not None else "")) \
        if isinstance(ag, dict) and ag.get("n") else NOT_RUN

    def table(head, rows):
        return ["| System | " + " | ".join(head) + " |", "|---|" + "---|" * len(head), *rows, ""]

    def cells(fn):
        return [fn(sys[n]) if sys[n] != NOT_RUN else NOT_RUN for n in names]
    out = [f"# Evaluation summary ({s['generated']})", "",
           "## How to read this", "",
           f"- {s['n_tasks']} tasks, one run per (system, task), **replayed** from frozen sources recorded live; LLM answers come from a content-addressed cache, "
           "so reruns are identical. Latency columns are wall-clock/recorded and not part of the determinism guarantee.",
           f"- Quality metrics are judged by `{s['judge']['model']}` (a different model from the pipeline's verifier). **Judge agreement vs hand labels: {ag_txt}.**",
           "- Tasks, intent labels and claim labels were written by the project author. Rates show mean [95% bootstrap CI over tasks]. One live recording, one replay: "
           "no run-to-run variance is measured.", ""]
    router_rows = [_row(n, [f"{sys[n]['routing']['accuracy'] * 100:.1f}%", f"{sys[n]['routing']['macro_f1'] * 100:.1f}%",
                            *(f"{sys[n]['routing']['latency_ms'][k]:.1f} ms" for k in ("p50", "p95", "p99"))] if sys[n] != NOT_RUN else [NOT_RUN] * 5) for n in names]
    out += ["## Router", "", *table(["workflow accuracy", "macro F1", "routing P50", "routing P95", "routing P99"], router_rows)]
    rk = [("completion", "task completion"), ("e2e_success", "end-to-end success"), ("unsupported_rate", "unsupported-claim rate"),
          ("citation_correctness", "citation correctness"), ("source_relevance", "source relevance")]
    out += ["## Research quality", "", *table([x[1] for x in rk], [_row(n, [(_f(sys[n]["research"].get(k), pct=True) if sys[n] != NOT_RUN else NOT_RUN) for k, _ in rk]) for n in names])]
    ak = [("tool_selection", "tool selection"), ("plan_success", "plan success"), ("recovery_success", "recovery success"), ("end_to_end_success", "end-to-end")]
    out += ["## Agent system", "", *table([x[1] for x in ak], [_row(n, [(_f(sys[n]["agent"].get(k), pct=True) if sys[n] != NOT_RUN else NOT_RUN) for k, _ in ak]) for n in names])]
    out += ["## Efficiency", "", *table(["LLM calls/run", "input tok/run", "output tok/run", "cost/run (USD)", "total cost (USD)", "tool calls/run", "wall P50", "wall P95"], [
        _row(n, ([_f(sys[n]["efficiency"]["llm_calls"]), f"{sys[n]['efficiency']['input_tokens']['mean']:.0f}", f"{sys[n]['efficiency']['output_tokens']['mean']:.0f}",
                  f"${sys[n]['efficiency']['cost_usd']['mean']:.5f}", f"${sys[n]['efficiency']['total_cost_usd']:.3f}", _f(sys[n]["efficiency"]["tool_calls"]),
                  f"{sys[n]['wall_ms']['p50'] / 1000:.1f} s", f"{sys[n]['wall_ms']['p95'] / 1000:.1f} s"] if sys[n] != NOT_RUN else [NOT_RUN] * 8)) for n in names])]
    if "D" in sys and sys["D"] != NOT_RUN:
        keys = ["completion", "e2e_success", "unsupported_rate", "citation_correctness"]
        rows = [_row("D (full harness)", [_f(sys["D"]["research"].get(k), pct=True) for k in keys] + ["-"])]
        for n, d in s["ablations"].items():
            a = s["ablation_systems"][n]
            if d == NOT_RUN or a == NOT_RUN:
                rows.append(_row(n, [NOT_RUN] * 5))
            else:
                rows.append(_row(n, [_f(a["research"].get(k), pct=True) for k in keys] + [", ".join(f"{k} {v * 100:+.1f}pp" for k, v in d["delta_vs_D"]["research"].items() if k in keys)]))
        out += ["## Harness ablations", "", *table(["completion", "e2e success", "unsupported rate", "citation correctness", "delta vs D"], rows)]
    else:
        out += ["## Harness ablations", "", NOT_RUN, ""]
    bk = []
    for n in names:
        if sys[n] != NOT_RUN:
            for k in ("normal", "ambiguous", "tempting"):
                v = sys[n]["by_kind"].get(k, {})
                bk.append(_row(f"{n} / {k}", [f"{v[m] * 100:.1f}%" if m in v else "n/a" for m in ("completion", "e2e_success", "unsupported_rate")]))
    out += ["## By task kind", "", *table(["completion", "e2e success", "unsupported rate"], bk)]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    setup_logging()
    load_dotenv()
    cfg = load_config(args.config)
    base = None
    if not args.offline and not args.no_judge:
        from research_os.llm.gemini_client import GeminiLLMClient
        from research_os.llm.models import provider_settings
        prov = provider_settings()
        base = GeminiLLMClient(model_for_role("eval_judge"), prov["api_key_env"], prov["timeout_s"] * 3)
    summary = build_summary(cfg, base, use_judge=not args.no_judge)
    out = Path(cfg["paths"]["results"])
    write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(render_summary_md(summary))
    log.info("wrote %s and %s", out / "summary.json", out / "summary.md")


if __name__ == "__main__":
    main()
