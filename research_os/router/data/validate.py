"""Dataset validation: fails loudly on any quality problem."""
from __future__ import annotations

import argparse
import logging
import re

import pandas as pd

from research_os.common import intents_from_cfg, load_config, setup_logging

log = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = re.sub(r"[^\w\s]", "", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def validate(df: pd.DataFrame, intents: list[str], balance_tol: float = 0.10) -> None:
    """Raise ValueError on empty text, unknown labels, duplicates or class imbalance."""
    if (df["text"].astype(str).str.strip() == "").any():
        raise ValueError("empty text found")
    bad = set(df["intent"]) - set(intents)
    if bad:
        raise ValueError(f"unknown labels: {sorted(bad)}")
    norm = df["text"].map(normalize)
    if norm.duplicated().any():
        dups = df.loc[norm.duplicated(keep=False), "text"].tolist()[:5]
        raise ValueError(f"duplicate or near-duplicate texts, e.g. {dups}")
    counts = df["intent"].value_counts()
    mean = counts.mean()
    if (counts.max() - mean) / mean > balance_tol or (mean - counts.min()) / mean > balance_tol:
        raise ValueError(f"label imbalance: {counts.to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(cfg["paths"]["raw"])
    validate(df, intents_from_cfg(cfg))
    log.info("dataset OK: %d rows, %s", len(df), df["intent"].value_counts().to_dict())


if __name__ == "__main__":
    main()
