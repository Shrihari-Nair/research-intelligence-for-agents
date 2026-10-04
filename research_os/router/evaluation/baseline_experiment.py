"""M6.1: baseline routing experiment (LLM-only vs MiniLM-only vs Hybrid) on the current domain."""
from __future__ import annotations

import argparse
import logging

import pandas as pd
import torch

from research_os.common import get_hardware_info, intents_from_cfg, load_config, set_seed, setup_logging, write_json
from research_os.router.data.load import load_split
from research_os.router.evaluation.compare_systems import NOT_RUN, run_routing_eval
from research_os.router.evaluation.evaluate import plot_confusion

log = logging.getLogger(__name__)


def run_experiment(systems: dict, df: pd.DataFrame, intents: list[str], pricing: dict, out_dir: str,
                   seed: int, split: str = "test", extra: dict | None = None) -> dict:
    """Run each system (None -> not_run) on `df`; write JSON, confusion PNG and CSV under `out_dir`."""
    res: dict = {"split": split, "seed": seed, "n": len(df), "hardware": get_hardware_info(),
                 "pricing_per_mtok_note": "configurable, not authoritative", "pricing_per_mtok": pricing,
                 **(extra or {}), "systems": {}}
    rows = []
    for name, system in systems.items():
        if system is None:
            res["systems"][name] = NOT_RUN
            rows.append({"system": name, "status": NOT_RUN})
            continue
        r = run_routing_eval(system, df, intents, pricing)
        res["systems"][name] = r
        rows.append({"system": name, "status": "ok", **{k: r[k] for k in (
            "accuracy", "macro_f1", "p50_ms", "p95_ms", "llm_calls", "input_tokens", "output_tokens",
            "cost_per_request")}})
    write_json(f"{out_dir}/baseline_results.json", res)
    pd.DataFrame(rows).to_csv(f"{out_dir}/routing_benchmark.csv", index=False)
    best = next((n for n in ("hybrid", "minilm_only") if isinstance(res["systems"].get(n), dict)), None)
    if best:
        plot_confusion(res["systems"][best]["confusion_matrix"], intents, f"{out_dir}/confusion_matrix.png",
                       f"{best} on {split}")
    return res


def main() -> None:
    from research_os.legacy.factory import build_llm, build_system
    from research_os.router.loader import load_router

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--routing", default="configs/routing.yaml")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    setup_logging()
    cfg, rcfg = load_config(args.config), load_config(args.routing)
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["benchmark"]["threads"])
    intents = intents_from_cfg(cfg)
    router, _, _ = load_router(cfg["paths"]["model"], cfg["paths"]["baseline_model"])
    llm, backend = build_llm(rcfg)
    systems = {"minilm_only": build_system("minilm", router, llm, rcfg),
               "hybrid": build_system("hybrid", router, llm, rcfg),
               "llm_only": build_system("llm", router, llm, rcfg) if backend == "gemini" else None}
    df = load_split(args.split, cfg["paths"]["processed"])
    extra = {"llm_backend": backend, "threshold": rcfg["threshold"], "torch_threads": torch.get_num_threads()}
    res = run_experiment(systems, df, intents, rcfg["pricing_per_mtok"], args.out, cfg["seed"], args.split, extra)
    for n, r in res["systems"].items():
        log.info("%s: %s", n, r if isinstance(r, str) else f"acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
