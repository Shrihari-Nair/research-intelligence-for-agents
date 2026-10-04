"""Load processed splits as DataFrames / HF datasets with integer labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset


def load_split(name: str, processed_dir: str = "data") -> pd.DataFrame:
    """Read `<dir>/<name>.jsonl` (or .csv). 'val' is accepted as an alias of 'validation'."""
    base = Path(processed_dir)
    for stem in (name, "validation" if name == "val" else None, "val" if name == "validation" else None):
        if stem is None:
            continue
        if (base / f"{stem}.jsonl").exists():
            return pd.read_json(base / f"{stem}.jsonl", lines=True)[["text", "intent"]]
        if (base / f"{stem}.csv").exists():
            return pd.read_csv(base / f"{stem}.csv")[["text", "intent"]]
    raise FileNotFoundError(f"no split file for {name!r} under {base} (.jsonl or .csv)")


def to_dataset(df: pd.DataFrame, intents: list[str]) -> Dataset:
    """HF Dataset with `text` and integer `label` (index in `intents`)."""
    idx = {n: i for i, n in enumerate(intents)}
    return Dataset.from_dict({"text": df["text"].tolist(),
                              "label": [idx[x] for x in df["intent"]]})
