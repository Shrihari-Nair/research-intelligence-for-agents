"""Fine-tune MiniLM with SetFit. Hyperparameters are chosen on val only; test is scored once."""
from __future__ import annotations

import argparse
import itertools
import logging
import time

import yaml
from setfit import SetFitModel, Trainer, TrainingArguments

from research_os.common import get_hardware_info, intents_from_cfg, load_config, set_seed, setup_logging, write_json
from research_os.router.data.load import load_split, to_dataset
from research_os.router.evaluation.metrics import compute_metrics
from research_os.router.model import IntentRouter

log = logging.getLogger(__name__)


def fit(cfg: dict, intents: list[str], train_df, seed: int, num_iterations: int, num_epochs: int):
    """Train one SetFit model; returns (model, train_seconds)."""
    s = cfg["setfit"]
    model = SetFitModel.from_pretrained(cfg["base_model"], labels=intents)
    args = TrainingArguments(batch_size=s["batch_size"], num_epochs=num_epochs,
                             num_iterations=num_iterations, body_learning_rate=s["learning_rate"],
                             seed=seed, show_progress_bar=False)
    trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train_df, intents))
    t0 = time.perf_counter()
    trainer.train()
    return model, time.perf_counter() - t0


def score(model: SetFitModel, df, intents: list[str]) -> dict:
    """Metrics of a model on a dataframe split."""
    preds = [r["intent"] for r in IntentRouter(model, intents).predict_batch(df["text"].tolist())]
    return compute_metrics(df["intent"].tolist(), preds, intents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-tune", action="store_true", help="skip val grid, use config values")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    intents = intents_from_cfg(cfg)
    proc = cfg["paths"]["processed"]
    train, val, test = load_split("train", proc), load_split("val", proc), load_split("test", proc)

    grid_log, best = [], (cfg["setfit"]["num_iterations"], cfg["setfit"]["num_epochs"])
    if not args.no_tune:
        best_f1 = -1.0
        for ni, ne in itertools.product(cfg["tuning"]["num_iterations"], cfg["tuning"]["num_epochs"]):
            set_seed(seed)
            model, secs = fit(cfg, intents, train, seed, ni, ne)
            m = score(model, val, intents)
            grid_log.append({"num_iterations": ni, "num_epochs": ne, "val_macro_f1": m["macro_f1"],
                             "val_accuracy": m["accuracy"], "train_seconds": secs})
            log.info("grid ni=%d ne=%d val macroF1=%.4f (%.0fs)", ni, ne, m["macro_f1"], secs)
            # prefer cheaper config on ties: strictly-greater only, grid iterates cheap -> costly
            if m["macro_f1"] > best_f1:
                best_f1, best = m["macro_f1"], (ni, ne)
        cfg["setfit"]["num_iterations"], cfg["setfit"]["num_epochs"] = best
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        raw["setfit"]["num_iterations"], raw["setfit"]["num_epochs"] = best
        with open(args.config, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

    set_seed(seed)
    model, secs = fit(cfg, intents, train, seed, *best)
    model.save_pretrained(cfg["paths"]["model"])  # explicit save, never rely on checkpoints
    val_m = score(model, val, intents)
    write_json(f"{cfg['paths']['results']}/train_log.json", {
        "train_seconds": secs, "num_iterations": best[0], "num_epochs": best[1], "seed": seed,
        "base_model": cfg["base_model"], "n_train": len(train), "val_metrics": val_m,
        "tuning_grid_on_val": grid_log, "hardware": get_hardware_info()})

    reloaded = IntentRouter.load(cfg["paths"]["model"], intents)
    preds = [r["intent"] for r in reloaded.predict_batch(test["text"].tolist())]
    m = compute_metrics(test["intent"].tolist(), preds, intents)
    m.update({"model": "setfit fine-tuned MiniLM", "split": "test", "seed": seed,
              "train_seconds": secs, "hardware": get_hardware_info()})
    write_json(f"{cfg['paths']['results']}/finetuned.json", m)
    log.info("finetuned test acc=%.4f macroF1=%.4f emergency_recall=%s", m["accuracy"],
             m["macro_f1"], m["emergency_recall"])


if __name__ == "__main__":
    main()
