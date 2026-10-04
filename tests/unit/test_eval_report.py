import json

import pytest

from evaluation.report_results import build_summary, judge_runs, render_summary_md
from evaluation.validate_dataset import Task
from research_os.llm.mock_client import ScriptedLLM

CFG = None


def mkcfg(tmp_path):
    return {"seed": 1, "systems": ["A", "D"], "ablations": ["D-no-verifier"],
            "thresholds": {"min_citation_correctness": 0.8, "max_unsupported_rate": 0.2}, "pricing_per_mtok": {"input": 1.0, "output": 2.0},
            "bootstrap": {"n": 100, "confidence": 0.95}, "judge": {"batch_size": 10, "max_evidence_per_run": 30, "max_sources_per_run": 10},
            "paths": {"tasks": str(tmp_path / "tasks.jsonl"), "gold_claims": str(tmp_path / "gold.jsonl"), "runs": str(tmp_path / "runs"),
                      "results": str(tmp_path / "res"), "judge_cache": str(tmp_path / "judge.jsonl"), "llm_cache": "x", "cassettes": "x"}}


def task(i):
    return Task(id=f"t{i}", input=f"a research question number {i}", expected_intent="TECHNOLOGY_COMPARISON",
                expected_workflow="technology_comparison", required_source_types=[], min_supported_claims=1, kind="normal")


def run(system, tid, labels_ok=True, status="SUCCESS"):
    ev = [{"claim_id": f"{tid}c{i}", "claim": "x", "quote": "q", "source_id": "s1", "source_title": "T", "source_type": "documentation", "verbatim": True}
          for i in range(2)]
    return {"system": system, "task_id": tid, "status": status, "workflow": "technology_comparison", "router_intent": "TECHNOLOGY_COMPARISON",
            "routing_mode": "DIRECT", "supervisor_llm_used": system == "A", "steps": 8, "recoveries": [], "failures": [], "halted_reason": None,
            "report": {"question": "a research question", "executive_summary": "s", "findings": [{"statement": "f", "claim_ids": [ev[0]["claim_id"]]}],
                       "evidence": [{"claim_id": e["claim_id"], "source_id": "s1"} for e in ev], "contradictions": []},
            "tools": {"calls": 3, "searches": 2, "by_name": {"web_search": 2, "fetch_url": 1}},
            "llm_totals": {"calls": 4, "input_tokens": 1000, "output_tokens": 500, "latency_ms": 100.0},
            "routing": {"classifier_ms": 5.0, "llm_calls": 0, "llm_latency_ms": 0.0}, "routing_llm_latency_ms": 0.0, "wall_ms": 100.0,
            "evidence_checks": ev, "cited_sources": [{"source_id": "s1", "title": "T", "source_type": "documentation", "snippet": "x"}]}


def write_runs(tmp_path, systems, n=3):
    for s in systems:
        d = tmp_path / "runs" / s
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"t{i}.json").write_text(json.dumps(run(s, f"t{i}")))
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(task(i).model_dump()) + "\n" for i in range(n)))


class AllGood(ScriptedLLM):
    """Judge that labels every claim SUPPORTED and every source 2."""

    def complete(self, prompt, **kw):
        import re

        from research_os.llm.client import LLMResponse
        if "citations in a research report" in prompt:
            ids = re.findall(r"<<<ITEM id=(\S+)", prompt)
            return LLMResponse(text=json.dumps({"labels": [{"id": i, "label": "SUPPORTED"} for i in ids]}))
        ids = re.findall(r"<<<SOURCE id=(\S+)", prompt)
        return LLMResponse(text=json.dumps({"ratings": [{"id": i, "score": 2} for i in ids]}))


def test_judge_runs_labels_every_claim_and_source_and_uses_the_cache(tmp_path):
    cfg = mkcfg(tmp_path)
    write_runs(tmp_path, ["D"], n=2)
    runs = [json.loads(p.read_text()) for p in sorted((tmp_path / "runs" / "D").glob("*.json"))]
    j = judge_runs(runs, cfg, AllGood([]))
    assert set(j) == {("D", "t0"), ("D", "t1")} and j[("D", "t0")]["claims"]["t0c0"] == "SUPPORTED" and j[("D", "t0")]["relevance"] == {"s1": 2}
    again = judge_runs(runs, cfg, None)                       # offline: all judge calls are served from the cache
    assert again == j


def test_summary_has_systems_ablation_deltas_and_marks_missing_systems_not_run(tmp_path):
    cfg = mkcfg(tmp_path)
    write_runs(tmp_path, ["A", "D"])
    s = build_summary(cfg, AllGood([]))
    assert set(s["systems"]) == {"A", "D"} and s["systems"]["A"]["n"] == 3
    assert s["systems"]["D"]["research"]["citation_correctness"]["mean"] == 1.0
    assert s["ablations"]["D-no-verifier"] == "not_run"
    assert s["judge"]["model"] and s["judge"]["agreement"] == "not_run"


def test_ablation_deltas_are_relative_to_D(tmp_path):
    cfg = mkcfg(tmp_path)
    write_runs(tmp_path, ["A", "D", "D-no-verifier"])
    s = build_summary(cfg, AllGood([]))
    d = s["ablations"]["D-no-verifier"]
    assert d["delta_vs_D"]["research"]["completion"] == pytest.approx(0.0) and "efficiency.llm_calls" in d["delta_vs_D"]["metrics"]


def test_judge_agreement_with_gold_is_reported_when_gold_exists(tmp_path):
    cfg = mkcfg(tmp_path)
    write_runs(tmp_path, ["D"])
    gold = [{"task_id": "t0", "claim_id": "t0c0", "label": "SUPPORTED"}, {"task_id": "t0", "claim_id": "t0c1", "label": "UNSUPPORTED"},
            {"task_id": "t1", "claim_id": "t1c0", "label": "SUPPORTED"}]
    (tmp_path / "gold.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gold))
    s = build_summary(cfg, AllGood([]))
    ag = s["judge"]["agreement"]
    assert ag["n"] == 3 and ag["accuracy"] == pytest.approx(2 / 3)


def test_markdown_renders_the_required_tables_and_flags_not_run(tmp_path):
    cfg = mkcfg(tmp_path)
    write_runs(tmp_path, ["A", "D"])
    md = render_summary_md(build_summary(cfg, AllGood([])))
    for h in ("## Router", "## Research quality", "## Agent system", "## Efficiency", "## Harness ablations", "## How to read this"):
        assert h in md, h
    assert "not_run" in md and "| A |" in md and "| D |" in md and "judge agreement" in md.lower()
