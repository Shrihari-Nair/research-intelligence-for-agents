"""Stratified 70/15/15 split with zero-overlap assertion."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from research_os.router.data.validate import normalize


def split(df: pd.DataFrame, seed: int, frac: tuple[float, float, float] = (0.70, 0.15, 0.15)
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test), stratified by intent; asserts no text overlap."""
    train, rest = train_test_split(df, train_size=frac[0], stratify=df["intent"],
                                   random_state=seed)
    val, test = train_test_split(rest, train_size=frac[1] / (frac[1] + frac[2]),
                                 stratify=rest["intent"], random_state=seed)
    sets = [set(map(normalize, s["text"])) for s in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "split leakage"
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))
