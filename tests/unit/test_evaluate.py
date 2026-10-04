from research_os.router.evaluation.banner import format_banner


def test_banner_matches_roadmap_format():
    out = format_banner({"model": "MiniLM", "fine_tuned": "Yes", "accuracy": 0.9712,
                         "macro_f1": 0.9655, "p50_ms": 5.4, "p95_ms": 8.1, "model_size_mb": 87.3,
                         "cpu_memory_mb": 412.0})
    expected = "\n".join([
        "=============================",
        "Intent Router Evaluation",
        "=============================",
        "Model: MiniLM",
        "Fine-tuned: Yes",
        "Accuracy:       97.12%",
        "Macro F1:       96.55%",
        "P50 latency:    5.4 ms",
        "P95 latency:    8.1 ms",
        "Model size:     87.3 MB",
        "CPU memory:     412.0 MB",
        "=============================",
    ])
    assert out == expected


def test_banner_marks_missing_fields_na():
    out = format_banner({"model": "MiniLM", "fine_tuned": "No", "accuracy": 0.5, "macro_f1": 0.4})
    assert "P50 latency:    n/a" in out and "CPU memory:     n/a" in out
