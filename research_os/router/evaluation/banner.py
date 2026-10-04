"""Human-readable summary banner (the one place printing is intended)."""
from __future__ import annotations

_BAR = "============================="


def _fmt(d: dict, key: str, kind: str) -> str:
    v = d.get(key)
    if v is None:
        return "n/a"
    return {"pct": f"{v * 100:.2f}%", "ms": f"{v:.1f} ms", "mb": f"{v:.1f} MB"}[kind]


def format_banner(d: dict) -> str:
    """Render the roadmap's banner; missing numeric fields show as n/a."""
    return "\n".join([
        _BAR, "Intent Router Evaluation", _BAR,
        f"Model: {d['model']}", f"Fine-tuned: {d['fine_tuned']}",
        f"Accuracy:       {_fmt(d, 'accuracy', 'pct')}",
        f"Macro F1:       {_fmt(d, 'macro_f1', 'pct')}",
        f"P50 latency:    {_fmt(d, 'p50_ms', 'ms')}",
        f"P95 latency:    {_fmt(d, 'p95_ms', 'ms')}",
        f"Model size:     {_fmt(d, 'model_size_mb', 'mb')}",
        f"CPU memory:     {_fmt(d, 'cpu_memory_mb', 'mb')}",
        _BAR,
    ])
