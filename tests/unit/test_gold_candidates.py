import json

from evaluation.export_gold_candidates import sample_candidates


def make(tmp_path, n_tasks=4, claims=6):
    runs = tmp_path / "runs" / "D"
    runs.mkdir(parents=True)
    tasks = []
    for i in range(n_tasks):
        tid = f"t{i}"
        tasks.append({"id": tid, "input": f"question number {i} about things", "expected_intent": "FACT_LOOKUP", "expected_workflow": "fact_lookup",
                      "required_source_types": [], "must_have_citations": True, "min_supported_claims": 1, "kind": "normal", "gold": i != 3, "notes": ""})
        (runs / f"{tid}.json").write_text(json.dumps({"evidence_checks": [
            {"claim_id": f"{tid}c{j}", "claim": f"claim {j}", "quote": "q", "source_title": "T", "source_type": "documentation", "verbatim": True}
            for j in range(claims)]}))
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return {"paths": {"tasks": str(tmp_path / "tasks.jsonl"), "runs": str(tmp_path / "runs")}}


def test_only_gold_tasks_capped_per_task_deterministic(tmp_path):
    cfg = make(tmp_path)
    a = sample_candidates(cfg, n=100, per_task=2, seed=1)
    assert a == sample_candidates(cfg, n=100, per_task=2, seed=1)
    assert len(a) == 6 and {c["task_id"] for c in a} == {"t0", "t1", "t2"}          # t3 is not gold
    assert all(sum(c["task_id"] == t for c in a) == 2 for t in ("t0", "t1", "t2"))
    assert len(sample_candidates(cfg, n=4, per_task=2, seed=1)) == 4
    assert [c["claim_id"] for c in a] != [c["claim_id"] for c in sample_candidates(cfg, n=100, per_task=2, seed=2)]


def test_missing_runs_are_skipped(tmp_path):
    cfg = make(tmp_path)
    (tmp_path / "runs" / "D" / "t1.json").unlink()
    assert {c["task_id"] for c in sample_candidates(cfg, n=100, per_task=2, seed=1)} == {"t0", "t2"}
