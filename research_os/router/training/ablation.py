"""Data-size ablation: N examples/intent x seeds, evaluated on test."""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from setfit import sample_dataset  # noqa: E402

from research_os.common import (  # noqa: E402
    get_hardware_info,
    intents_from_cfg,
    load_config,
    set_seed,
    setup_logging,
    write_json,
)
from research_os.router.data.load import load_split, to_dataset  # noqa: E402
from research_os.router.training.train_setfit import fit, score  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    intents = intents_from_cfg(cfg)
    train, test = load_split("train", cfg["paths"]["processed"]), load_split("test", cfg["paths"]["processed"])
    sizes = args.sizes or cfg["ablation"]["sizes"]
    per_class_avail = int(train["intent"].value_counts().min())
    results = {}
    for n in sizes:
        eff = min(n, per_class_avail)
        runs = []
        for seed in cfg["ablation"]["seeds"]:
            set_seed(seed)
            sub = sample_dataset(to_dataset(train, intents), label_column="label",
                                 num_samples=eff, seed=seed)
            import pandas as pd
            sub_df = pd.DataFrame({"text": sub["text"], "intent": [intents[i] for i in sub["label"]]})
            model, secs = fit(cfg, intents, sub_df, seed, cfg["setfit"]["num_iterations"],
                              cfg["setfit"]["num_epochs"])
            m = score(model, test, intents)
            runs.append({"seed": seed, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "emergency_recall": m["emergency_recall"], "train_seconds": secs})
            log.info("n=%d (eff %d) seed=%d acc=%.4f f1=%.4f", n, eff, seed, m["accuracy"], m["macro_f1"])
        agg = {}
        for k in ("accuracy", "macro_f1", "emergency_recall", "train_seconds"):
            v = np.array([r[k] for r in runs], dtype=float)
            agg[f"{k}_mean"], agg[f"{k}_std"] = float(v.mean()), float(v.std())
        results[str(n)] = {"effective_per_intent": eff, **agg, "per_seed": runs}
    write_json(f"{cfg['paths']['results']}/ablation.json", {"results": results, "setfit": cfg["setfit"],
                                                            "hardware": get_hardware_info()})
    xs = [int(k) for k in results]
    plt.figure(figsize=(5, 3.5))
    for key, label in (("macro_f1", "macro F1"), ("accuracy", "accuracy")):
        plt.errorbar(xs, [results[str(x)][f"{key}_mean"] for x in xs],
                     yerr=[results[str(x)][f"{key}_std"] for x in xs], marker="o", capsize=3, label=label)
    plt.xlabel("training examples per intent")
    plt.ylabel("score on test set")
    plt.title("SetFit data-size ablation (mean ± std, 3 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg['paths']['results']}/ablation.png", dpi=150)


if __name__ == "__main__":
    set_seed(42)
    main()
