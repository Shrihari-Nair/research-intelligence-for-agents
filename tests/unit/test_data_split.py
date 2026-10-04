import pandas as pd
import pytest

from research_os.common import load_intents
from research_os.router.data.generate import generate
from research_os.router.data.split import split
from research_os.router.data.validate import normalize

PROC = "data/healthcare/processed"


def test_split_proportions_and_no_overlap():
    df = generate(42, 40)
    tr, va, te = split(df, 42)
    assert len(tr) + len(va) + len(te) == len(df)
    sets = [set(map(normalize, s["text"])) for s in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for name, part, frac in (("train", tr, .7), ("val", va, .15), ("test", te, .15)):
        per = part["intent"].value_counts()
        assert per.max() - per.min() <= 1, name
        assert abs(len(part) / len(df) - frac) < 0.02


def _load(name):
    try:
        return pd.read_csv(f"{PROC}/{name}.csv")
    except FileNotFoundError:
        pytest.skip("run `make data` first")


def test_processed_files_leak_free_incl_hard_test():
    parts = {n: set(map(normalize, _load(n)["text"])) for n in ("train", "val", "test", "hard_test")}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (parts[a] & parts[b]), f"leak between {a} and {b}"


def test_hard_test_labels_valid_and_covered():
    h = _load("hard_test")
    assert set(h["intent"]) <= set(load_intents("configs/healthcare/intents.yaml"))
    assert 40 <= len(h) <= 70
    assert (h["intent"].value_counts() >= 5).all()
