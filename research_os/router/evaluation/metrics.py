"""Shared classification metrics."""
from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: list[str], y_pred: list[str], intents: list[str],
                    emergency_label: str = "EMERGENCY") -> dict:
    """Accuracy, macro P/R/F1, per-intent F1, confusion matrix and EMERGENCY recall."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=intents, average="macro",
                                                 zero_division=0)
    _, rec, f1s, sup = precision_recall_fscore_support(y_true, y_pred, labels=intents,
                                                       zero_division=0)
    per_f1 = dict(zip(intents, map(float, f1s), strict=True))
    recalls = dict(zip(intents, map(float, rec), strict=True))
    support = dict(zip(intents, map(int, sup), strict=True))
    em_recall = recalls[emergency_label] if support.get(emergency_label, 0) > 0 else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "per_intent_f1": per_f1,
        "emergency_recall": em_recall,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=intents).tolist(),
        "n": len(y_true),
    }
