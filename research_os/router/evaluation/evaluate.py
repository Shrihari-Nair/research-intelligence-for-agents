"""Evaluate a router (baseline or fine-tuned) on a split; writes JSON + confusion PNG."""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from research_os.common import get_hardware_info, intents_from_cfg, load_config, setup_logging, write_json  # noqa: E402
from research_os.router.data.load import load_split  # noqa: E402
from research_os.router.evaluation.banner import format_banner  # noqa: E402
from research_os.router.evaluation.metrics import compute_metrics  # noqa: E402
from research_os.router.loader import load_router  # noqa: E402

log = logging.getLogger(__name__)


def plot_confusion(cm: list[list[int]], intents: list[str], path: str, title: str) -> None:
    """Save a labeled confusion matrix PNG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(intents)), intents, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(intents)), intents, fontsize=7)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            dark = v >= max(map(max, cm)) / 2
            ax.text(j, i, v, ha="center", va="center", fontsize=7, color="white" if dark else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--model", required=True, help="'baseline' or path to a saved model")
    ap.add_argument("--split", default="test", choices=["val", "test", "hard_test", "independent_test"])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    intents = intents_from_cfg(cfg)
    router, name, tuned = load_router(args.model, cfg["paths"]["baseline_model"])
    df = load_split(args.split, cfg["paths"]["processed"])
    preds = [r["intent"] for r in router.predict_batch(df["text"].tolist())]
    m = compute_metrics(df["intent"].tolist(), preds, intents)
    m.update({"model": name, "fine_tuned": tuned, "split": args.split, "hardware": get_hardware_info()})
    res = cfg["paths"]["results"]
    tag = f"{name}_{args.split}"
    write_json(f"{res}/eval_{tag}.json", m)
    plot_confusion(m["confusion_matrix"], intents, f"{res}/confusion_{tag}.png", f"{name} on {args.split}")
    print(format_banner({"model": f"MiniLM ({name})", "fine_tuned": "Yes" if tuned else "No", **m}))
    if m.get("emergency_recall") is not None:
        print(f"EMERGENCY recall: {m['emergency_recall']}")


if __name__ == "__main__":
    main()
