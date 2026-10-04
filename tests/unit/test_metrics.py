import pytest

from research_os.router.evaluation.metrics import compute_metrics

INTENTS = ["A", "B", "EMERGENCY"]


def test_known_values():
    y_true = ["A", "A", "B", "B", "EMERGENCY", "EMERGENCY"]
    y_pred = ["A", "B", "B", "B", "EMERGENCY", "A"]
    m = compute_metrics(y_true, y_pred, INTENTS)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["emergency_recall"] == pytest.approx(0.5)
    assert m["per_intent_f1"]["B"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["confusion_matrix"][2][0] == 1  # EMERGENCY predicted as A


def test_emergency_recall_none_when_no_emergency_rows():
    m = compute_metrics(["A", "B"], ["A", "B"], INTENTS)
    assert m["emergency_recall"] is None
