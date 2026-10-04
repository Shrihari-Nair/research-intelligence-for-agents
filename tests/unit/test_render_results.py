import json

from scripts.render_results import render, update_readme


def w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def test_missing_files_render_tbd_not_invented_numbers(tmp_path):
    md = render(str(tmp_path))
    assert "TBD" in md and "%" not in md.replace("TBD", "")


def test_router_numbers_come_from_the_json_files(tmp_path):
    r = tmp_path / "router"
    w(r / "baseline.json", {"accuracy": 0.9074, "macro_f1": 0.9065})
    w(r / "finetuned.json", {"accuracy": 0.9753, "macro_f1": 0.9754, "hardware": {"cpu_model": "TestCPU", "cores": 8, "ram_gb": 16.0, "torch_version": "2.x"}})
    w(r / "eval_baseline_hard_test.json", {"accuracy": 0.9365, "macro_f1": 0.9396})
    w(r / "eval_router_hard_test.json", {"accuracy": 0.9365, "macro_f1": 0.9365})
    w(r / "eval_baseline_independent_test.json", {"accuracy": 0.8558, "macro_f1": 0.8474})
    w(r / "eval_router_independent_test.json", {"accuracy": 0.9615, "macro_f1": 0.9614})
    w(r / "bench_router.json", {"p50_ms": 11.0, "p95_ms": 15.0, "p99_ms": 16.0, "throughput_rps": 90.0, "model_size_mb": 91.6, "cpu_memory_mb": 570.0, "threads": 4})
    md = render(str(tmp_path))
    for needle in ("97.53%", "90.74%", "93.65%", "96.15%", "85.58%", "11.0 ms", "TestCPU", "91.6 MB"):
        assert needle in md, needle


def test_ablation_table_and_threshold(tmp_path):
    r = tmp_path / "router"
    w(r / "ablation.json", {"results": {"10": {"accuracy_mean": 0.82, "accuracy_std": 0.02, "macro_f1_mean": 0.81, "macro_f1_std": 0.02, "effective_per_intent": 10}}})
    w(r / "threshold_sweep.json", {"chosen_threshold": 0.6, "rule": "smallest threshold with covered accuracy >= 1.0 on val"})
    md = render(str(tmp_path))
    assert "82.00 ± 2.00" in md and "**0.6**" in md


def test_system_evaluation_and_failure_injection_are_embedded(tmp_path):
    w(tmp_path / "results" / "summary.md", "# Evaluation summary (2026)\n\n## Router\n\n| System | x |\n|---|---|\n| A | 1 |\n")
    w(tmp_path / "results" / "failure_injection.md", "# Failure injection\n\nnote\n\n| System | met |\n|---|---|\n| D | 100% |\n")
    w(tmp_path / "results" / "verifier_eval.json", {"accuracy": 1.0, "n": 30, "judge_model": "gemini-2.5-flash"})
    md = render(str(tmp_path))
    import re
    assert "| A | 1 |" in md and not re.search(r"^#{1,2} Evaluation summary", md, re.M) and "### Evaluation summary" in md   # headings are demoted
    assert "| D | 100% |" in md and "30 pairs" in md and "gemini-2.5-flash" in md


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nb\n")
    update_readme(str(p), "NEW")
    assert p.read_text() == "a\n<!-- RESULTS:START -->\nNEW\n<!-- RESULTS:END -->\nb\n"
