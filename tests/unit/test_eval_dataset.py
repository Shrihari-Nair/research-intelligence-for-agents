import json
from pathlib import Path

import pytest

from evaluation.validate_dataset import Task, load_tasks, near_duplicate_of, validate_dataset
from research_os.common import load_intents

TASKS = Path("evaluation/dataset/tasks.jsonl")


def mk(**kw):
    d = dict(id="t1", input="Compare A and B for something real", expected_intent="TECHNOLOGY_COMPARISON",
             expected_workflow="technology_comparison", required_source_types=["documentation"], must_have_citations=True,
             min_supported_claims=2, kind="normal", gold=False, notes="")
    d.update(kw)
    return d


def write(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_committed_dataset_is_valid_balanced_and_independent_of_router_data():
    rep = validate_dataset(TASKS, "data", require_cassettes=False)
    assert rep["n"] == 120 and rep["per_intent"] == {i: 15 for i in load_intents()}
    assert rep["kinds"] == {"normal": 72, "ambiguous": 24, "tempting": 24} and rep["gold"] == 32
    assert rep["router_overlap"] == []


def test_every_task_cites_a_known_intent_workflow_pair():
    for t in load_tasks(TASKS):
        assert t.expected_intent in load_intents() and t.expected_workflow == t.expected_intent.lower()


def test_schema_rejects_bad_intent_workflow_mismatch_and_bad_kind():
    with pytest.raises(ValueError):
        Task(**mk(expected_intent="NOPE"))
    with pytest.raises(ValueError, match="workflow"):
        Task(**mk(expected_workflow="market_research"))
    with pytest.raises(ValueError):
        Task(**mk(kind="weird"))
    with pytest.raises(ValueError):
        Task(**mk(min_supported_claims=0))


def test_validator_flags_duplicates_imbalance_and_router_overlap(tmp_path):
    rows = [mk(id="a", input="Compare A and B for something real"), mk(id="b", input="compare a and b for something real!")]
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(write(tmp_path, rows), "data", require_cassettes=False, balance_tol=10)
    leak = mk(id="c", input=json.loads(Path("data/train.jsonl").read_text().splitlines()[0])["text"])
    rep = validate_dataset(write(tmp_path, [leak]), "data", require_cassettes=False, balance_tol=10, fail_on_overlap=False)
    assert rep["router_overlap"] == ["c"]
    with pytest.raises(ValueError, match="overlap"):
        validate_dataset(write(tmp_path, [leak]), "data", require_cassettes=False, balance_tol=10)


def test_missing_cassettes_fail_only_when_required(tmp_path):
    p = write(tmp_path, [mk(input="A genuinely novel evaluation question about widgets and gadgets")])
    validate_dataset(p, "data", require_cassettes=False, balance_tol=10)
    with pytest.raises(ValueError, match="cassette"):
        validate_dataset(p, "data", require_cassettes=True, cassette_dir=str(tmp_path / "cas"), balance_tol=10)
    (tmp_path / "cas").mkdir()
    (tmp_path / "cas" / "t1.jsonl").write_text("")
    (tmp_path / "cas" / "t1.sources.json").write_text("[]")
    validate_dataset(p, "data", require_cassettes=True, cassette_dir=str(tmp_path / "cas"), balance_tol=10)


def test_near_duplicate_detection_is_token_based():
    assert near_duplicate_of("compare postgres and mysql for oltp", ["Compare Postgres and MySQL for OLTP workloads"], 0.7)
    assert not near_duplicate_of("explain raft leader election", ["compare postgres and mysql"], 0.7)
