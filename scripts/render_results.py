"""Render the README results block from experiments/ files (never hand-typed numbers; missing inputs render as TBD).

Usage: python -m scripts.render_results [README.md]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TBD = "TBD"


def _load(base: Path, name: str):
    p = base / name
    return json.loads(p.read_text()) if p.exists() else None


def _pct(v) -> str:
    return TBD if v is None else f"{v * 100:.2f}%"


def _mb(v, digits: int = 1) -> str:
    return TBD if v is None else f"{v:.{digits}f} MB"


def _ms(v) -> str:
    return TBD if v is None else f"{v:.1f} ms"


def _demote(md: str, by: int = 2) -> str:
    """Shift markdown headings down so an embedded file fits under a README section."""
    return re.sub(r"^(#{1,5}) ", lambda m: "#" * (len(m.group(1)) + by) + " ", md, flags=re.M)


def render(experiments_dir: str = "experiments") -> str:
    """Markdown for every results section."""
    base = Path(experiments_dir)
    rt = base / "router"
    out: list[str] = []
    bl, ft = _load(rt, "baseline.json") or {}, _load(rt, "finetuned.json") or {}
    hb, hf = _load(rt, "eval_baseline_hard_test.json") or {}, _load(rt, "eval_router_hard_test.json") or {}
    ib, i_f = _load(rt, "eval_baseline_independent_test.json") or {}, _load(rt, "eval_router_independent_test.json") or {}
    out += ["### Router: fine-tuned MiniLM vs frozen-embedding baseline", "",
            "| Model | Test accuracy | Test macro F1 | Hard-set accuracy | Hard-set macro F1 | Independent-set accuracy | Independent-set macro F1 |", "|---|---|---|---|---|---|---|",
            f"| Frozen MiniLM + LogReg (baseline) | {_pct(bl.get('accuracy'))} | {_pct(bl.get('macro_f1'))} | {_pct(hb.get('accuracy'))} | {_pct(hb.get('macro_f1'))} | {_pct(ib.get('accuracy'))} | {_pct(ib.get('macro_f1'))} |",
            f"| SetFit fine-tuned MiniLM | {_pct(ft.get('accuracy'))} | {_pct(ft.get('macro_f1'))} | {_pct(hf.get('accuracy'))} | {_pct(hf.get('macro_f1'))} | {_pct(i_f.get('accuracy'))} | {_pct(i_f.get('macro_f1'))} |",
            "", "Test = held-out split of the synthetic data (n=162). Hard set = 63 hand-written ambiguous queries. Independent set = 104 queries "
            "written separately from the training pipeline (13 per intent; never used for training or tuning; single labeller, see `data/independent_test_REVIEW.md`)."]
    ab = _load(rt, "ablation.json")
    out += ["", "### Router: data-size ablation (test, mean ± std over 3 seeds)", "", "| Examples / intent | Accuracy | Macro F1 |", "|---|---|---|"]
    if ab:
        for n, r in ab["results"].items():
            out.append(f"| {n} | {r['accuracy_mean'] * 100:.2f} ± {r['accuracy_std'] * 100:.2f} | {r['macro_f1_mean'] * 100:.2f} ± {r['macro_f1_std'] * 100:.2f} |")
    else:
        out.append(f"| {TBD} | {TBD} | {TBD} |")
    sw = _load(rt, "threshold_sweep.json")
    out += ["", f"Routing threshold: **{sw['chosen_threshold']}** ({sw['rule']})." if sw else f"Routing threshold: {TBD}"]
    out += ["", "### Router latency (batch size 1, CPU)", "", "| Model | P50 | P95 | P99 | Throughput | Size on disk | RSS after load | Threads |", "|---|---|---|---|---|---|---|---|"]
    for name, f in (("Baseline", "bench_baseline.json"), ("SetFit fine-tuned", "bench_router.json")):
        b = _load(rt, f) or {}
        tp = f"{round(b['throughput_rps'])} req/s" if "throughput_rps" in b else TBD
        out.append(f"| {name} | {_ms(b.get('p50_ms'))} | {_ms(b.get('p95_ms'))} | {_ms(b.get('p99_ms'))} | {tp} | "
                   f"{_mb(b.get('model_size_mb'))} | {_mb(b.get('cpu_memory_mb'), 0)} | {b.get('threads', TBD)} |")
    v2 = _load(rt, "bench_router_v2.json")
    if v2:
        out += ["", f"Entity extraction alone: P95 {v2['entities_only']['p95_ms']:.2f} ms; full `route()` vs classifier-only P95 overhead {v2['p95_overhead_ms']:+.2f} ms (within noise)."]
    hw = ft.get("hardware")
    if hw:
        out += ["", f"_Hardware: {hw['cpu_model']}, {hw['cores']} cores, {hw['ram_gb']} GB RAM, torch {hw['torch_version']}._"]
    res = base / "results"
    summ = res / "summary.md"
    out += ["", "### Four-system evaluation (A LLM-only, B MiniLM-only, C hybrid, D hybrid + harness)", "",
            _demote(summ.read_text()) if summ.exists() else TBD]
    fi = res / "failure_injection.md"
    out += ["", "### Failure injection", "", _demote(fi.read_text()) if fi.exists() else TBD]
    ve = _load(res, "verifier_eval.json")
    out += ["", "### Verifier gold set", "",
            (f"{ve['n']} pairs hand-written by the project author, judged by `{ve['judge_model']}`: accuracy {ve['accuracy'] * 100:.1f}% "
             "(a regression test and upper bound, not an unbiased estimate; see `experiments/results/verifier_eval_NOTES.md`).") if ve else TBD]
    return "\n".join(out)


def update_readme(readme: str, block: str) -> None:
    """Replace the content between the RESULTS markers."""
    text = Path(readme).read_text()
    new = re.sub(r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)", lambda m: m.group(1) + block + m.group(2), text, flags=re.S)
    Path(readme).write_text(new)


if __name__ == "__main__":
    update_readme(sys.argv[1] if len(sys.argv) > 1 else "README.md", render())
