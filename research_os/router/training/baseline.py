"""Baseline: pretrained MiniLM (no fine-tuning) + LogisticRegression on frozen embeddings."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from research_os.common import get_hardware_info, intents_from_cfg, load_config, set_seed, setup_logging, write_json
from research_os.router.baseline_router import BaselineRouter
from research_os.router.data.load import load_split
from research_os.router.evaluation.metrics import compute_metrics

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    intents = intents_from_cfg(cfg)
    idx = {n: i for i, n in enumerate(intents)}
    train, test = load_split("train", cfg["paths"]["processed"]), load_split("test", cfg["paths"]["processed"])

    enc = SentenceTransformer(cfg["base_model"], device="cpu")
    t0 = time.perf_counter()
    head = LogisticRegression(max_iter=1000, random_state=seed)
    head.fit(enc.encode(train["text"].tolist(), show_progress_bar=False),
             [idx[x] for x in train["intent"]])
    train_seconds = time.perf_counter() - t0

    out = Path(cfg["paths"]["baseline_model"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"encoder": cfg["base_model"], "head": head}, out / "head.joblib")

    router = BaselineRouter(enc, head, intents)
    preds = [r["intent"] for r in router.predict_batch(test["text"].tolist())]
    m = compute_metrics(test["intent"].tolist(), preds, intents)
    m.update({"model": "baseline (frozen MiniLM + LogReg)", "split": "test", "seed": seed,
              "head_train_seconds": train_seconds, "hardware": get_hardware_info()})
    write_json(f"{cfg['paths']['results']}/baseline.json", m)
    log.info("baseline test acc=%.4f macroF1=%.4f emergency_recall=%s", m["accuracy"], m["macro_f1"],
             m["emergency_recall"])


if __name__ == "__main__":
    main()
