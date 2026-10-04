"""M1 smoke test: prove SetFit fine-tuning + save/load works on CPU (SST-2)."""
from __future__ import annotations

import argparse
import inspect
import logging
import tempfile
import time

import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

from research_os.common import get_hardware_info, load_config, set_seed, setup_logging, write_json

log = logging.getLogger(__name__)
LABELS = ["negative", "positive"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    ds = load_dataset("stanfordnlp/sst2")
    train = sample_dataset(ds["train"], label_column="label", num_samples=8, seed=seed)
    train = train.rename_column("sentence", "text") if "sentence" in train.column_names else train
    ev = ds["validation"].shuffle(seed=seed).select(range(200))
    ev = ev.rename_column("sentence", "text") if "sentence" in ev.column_names else ev

    model = SetFitModel.from_pretrained(cfg["base_model"], labels=LABELS)
    targs = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=10, seed=seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=ev)
    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0
    acc = trainer.evaluate(ev)["accuracy"]

    texts = list(ev["text"][:20])
    proba = np.asarray(model.predict_proba(texts))
    preds = model.predict(texts)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = SetFitModel.from_pretrained(d)
        same_preds = list(reloaded.predict(texts)) == list(preds)
        same_proba = bool(np.allclose(np.asarray(reloaded.predict_proba(texts)), proba, atol=1e-5))
    assert same_preds and same_proba, "save/load round trip changed predictions"

    api = {
        "predict_proba_shape": list(proba.shape),
        "predict_returns_label_strings": isinstance(preds[0], str),
        "predict_batch_size_arg": "batch_size" in inspect.signature(model.predict).parameters,
        "has_eval_strategy": "eval_strategy" in inspect.signature(TrainingArguments).parameters,
        "round_trip_identical": same_preds and same_proba,
    }
    out = {"accuracy": acc, "train_seconds": train_seconds, "seed": seed,
           "api_checks": api, "hardware": get_hardware_info()}
    write_json("results/smoke_sst2.json", out)
    log.info("accuracy=%.3f train_seconds=%.1f api=%s", acc, train_seconds, api)


if __name__ == "__main__":
    main()
