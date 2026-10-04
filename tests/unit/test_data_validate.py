import pandas as pd
import pytest

from research_os.router.data.validate import normalize, validate

INTENTS = ["A", "B"]


def df(rows):
    return pd.DataFrame(rows, columns=["text", "intent"])


def good():
    return df([("book me in", "A"), ("cancel it please", "B"),
               ("see the doctor", "A"), ("never mind cancel", "B")])


def test_valid_dataset_passes():
    validate(good(), INTENTS)


def test_normalize_ignores_case_punct_space():
    assert normalize("  Hello,   World! ") == "hello world"


def test_empty_text_fails():
    with pytest.raises(ValueError, match="empty"):
        validate(df([("  ", "A"), ("x y", "B")]), INTENTS)


def test_unknown_label_fails():
    with pytest.raises(ValueError, match="unknown"):
        validate(df([("hi there", "Z"), ("yo", "B")]), INTENTS)


def test_exact_duplicate_fails():
    with pytest.raises(ValueError, match="duplicate"):
        validate(df([("same text", "A"), ("same text", "A"), ("b", "B")]), INTENTS)


def test_near_duplicate_fails():
    with pytest.raises(ValueError, match="duplicate"):
        validate(df([("Cancel it!", "A"), ("cancel it", "B")]), INTENTS)


def test_imbalance_fails():
    rows = [(f"a{i} text", "A") for i in range(10)] + [("b one", "B")]
    with pytest.raises(ValueError, match="imbalance"):
        validate(df(rows), INTENTS)
