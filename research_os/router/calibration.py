"""Threshold sweep on validation data; chooses the routing threshold from val only."""
from __future__ import annotations

import argparse
import logging

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research_os.common import intents_from_cfg, load_config, setup_logging, write_json  # noqa: E402
from research_os.router.data.load import load_split  # noqa: E402
from research_os.router.model import IntentRouter  # noqa: E402

log = logging.getLogger(__name__)


def sweep_thresholds(conf: list[float], correct: list[bool], thresholds: list[float]) -> list[dict]:
    """Coverage and covered-subset accuracy at each threshold."""
    c, ok = np.asarray(conf), np.asarray(correct, dtype=bool)
    rows = []
    for t in thresholds:
        mask = c >= t
        n = int(mask.sum())
        rows.append({"threshold": float(t), "coverage": float(n / len(c)),
                     "covered_accuracy": float(ok[mask].mean()) if n else None, "n_covered": n})
    return rows


def choose_threshold(sweep: list[dict], target: float) -> tuple[float, str]:
    """Smallest threshold whose covered accuracy >= target; else best covered accuracy."""
    for r in sorted(sweep, key=lambda r: r["threshold"]):
        if r["covered_accuracy"] is not None and r["covered_accuracy"] >= target:
            return r["threshold"], f"smallest threshold with covered accuracy >= {target} on val"
    valid = [r for r in sweep if r["covered_accuracy"] is not None]
    best = max(valid, key=lambda r: (r["covered_accuracy"], -r["threshold"]))
    return best["threshold"], (f"no threshold reached covered accuracy >= {target} on val; "
                               "chose the one with highest covered accuracy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--routing", default="configs/routing.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg, rcfg = load_config(args.config), load_config(args.routing)
    intents = intents_from_cfg(cfg)
    router = IntentRouter.load(args.model or cfg["paths"]["model"], intents)
    val = load_split("val", cfg["paths"]["processed"])
    recs = router.predict_batch(val["text"].tolist())
    conf = [r["confidence"] for r in recs]
    correct = [r["intent"] == y for r, y in zip(recs, val["intent"], strict=True)]
    s = rcfg["sweep"]
    ths = [round(x, 4) for x in np.arange(s["low"], s["high"] + 1e-9, s["step"])]
    sweep = sweep_thresholds(conf, correct, ths)
    thr, rule = choose_threshold(sweep, rcfg["target_covered_accuracy"])
    write_json(f"{cfg['paths']['results']}/threshold_sweep.json",
               {"split": "val", "sweep": sweep, "chosen_threshold": thr, "rule": rule,
                "target_covered_accuracy": rcfg["target_covered_accuracy"]})
    rcfg["threshold"], rcfg["threshold_rule"] = thr, rule
    with open(args.routing, "w") as f:
        yaml.safe_dump(rcfg, f, sort_keys=False)
    xs = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(xs, [r["coverage"] for r in sweep], marker="o", label="coverage (handled by MiniLM)")
    ax.plot(xs, [r["covered_accuracy"] if r["covered_accuracy"] is not None else np.nan for r in sweep],
            marker="s", label="accuracy on covered")
    ax.axvline(thr, ls="--", color="gray", label=f"chosen = {thr:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_title("Coverage vs accuracy (validation)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{cfg['paths']['results']}/threshold_sweep.png", dpi=150)
    log.info("chosen threshold=%.2f (%s)", thr, rule)


if __name__ == "__main__":
    main()
