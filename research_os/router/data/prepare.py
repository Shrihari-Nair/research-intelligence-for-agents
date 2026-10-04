"""Generate -> validate -> split -> write data/processed/*.csv."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from research_os.common import intents_from_cfg, load_config, set_seed, setup_logging, write_json
from research_os.router.data.generate import generate
from research_os.router.data.split import split
from research_os.router.data.validate import normalize, validate

log = logging.getLogger(__name__)


def prepare_research(cfg: dict, intents: list[str]) -> None:
    """Research domain: templates + LLM rows -> data/{train,validation,test}.jsonl (+ stats)."""
    from research_os.router.data.build import build_dataset

    res = build_dataset(cfg, intents)
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    for name, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        res[key].to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        log.info("%s: %d rows", name, len(res[key]))
    write_json(f"{cfg['paths']['results']}/dataset_stats.json", res["stats"])
    log.info("hard_test (hand-written, untouched): %s", cfg["data"]["hard_test"])


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
    if cfg["data"].get("domain") == "research":
        return prepare_research(cfg, intents)

    df = generate(seed, cfg["data"]["per_intent"])
    validate(df, intents)
    hard = pd.read_csv(cfg["data"]["hard_test"])
    if set(hard["intent"]) - set(intents):
        raise ValueError("hard_test has unknown labels")
    # keep hard_test out of train/val/test
    hard_norm = set(map(normalize, hard["text"]))
    df = df[~df["text"].map(normalize).isin(hard_norm)].reset_index(drop=True)
    Path(cfg["paths"]["raw"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["paths"]["raw"], index=False)

    tr, va, te = split(df, seed, (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]))
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("val", va), ("test", te), ("hard_test", hard)):
        part.to_csv(out / f"{name}.csv", index=False)
        log.info("%s: %d rows", name, len(part))


if __name__ == "__main__":
    main()
