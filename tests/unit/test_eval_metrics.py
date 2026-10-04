import pytest

from evaluation.metrics import aggregate, bootstrap_ci, completion, expected_tools, score_run
from evaluation.validate_dataset import Task

CFG = {"thresholds": {"min_citation_correctness": 0.8, "max_unsupported_rate": 0.2}, "pricing_per_mtok": {"input": 1.0, "output": 2.0},
       "bootstrap": {"n": 200, "confidence": 0.95}, "seed": 1}


def task(**kw):
    d = dict(id="t1", input="a question", expected_intent="TECHNOLOGY_COMPARISON", expected_workflow="technology_comparison",
             required_source_types=["documentation"], min_supported_claims=2, kind="normal")
    d.update(kw)
    return Task(**d)


def run(**kw):
    ev = [{"claim_id": f"c{i}", "claim": "x", "quote": "q", "source_id": "s1", "source_title": "T", "source_type": "documentation", "verbatim": True}
          for i in range(3)]
    d = {"system": "D", "task_id": "t1", "status": "SUCCESS", "workflow": "technology_comparison", "router_intent": "TECHNOLOGY_COMPARISON",
         "routing_mode": "DIRECT", "supervisor_llm_used": False, "steps": 8, "recoveries": [], "failures": [], "halted_reason": None,
         "report": {"executive_summary": "s", "findings": [{"statement": "f", "claim_ids": ["c0"]}], "evidence": [{"claim_id": e["claim_id"],
                    "source_id": "s1"} for e in ev]}, "tools": {"calls": 4, "searches": 2, "by_name": {"web_search": 2, "fetch_url": 2}},
         "llm_totals": {"calls": 5, "input_tokens": 1_000_000, "output_tokens": 500_000, "latency_ms": 4000.0},
         "routing": {"classifier_ms": 10.0, "llm_calls": 0, "llm_latency_ms": 0.0}, "routing_llm_latency_ms": 0.0, "wall_ms": 1234.0,
         "evidence_checks": ev, "cited_sources": [{"source_id": "s1", "title": "T", "source_type": "documentation", "snippet": "x"}]}
    d.update(kw)
    return d


def test_completion_requires_success_sections_required_types_and_minimum_claims():
    assert completion(run(), task())["ok"] is True
    assert completion(run(status="PARTIAL"), task())["ok"] is False
    assert completion(run(report={"executive_summary": "", "findings": [], "evidence": []}), task())["sections"] is False
    blog = run()
    for e in blog["evidence_checks"]:
        e["source_type"] = "blog"
    assert completion(blog, task())["source_types"] is False and completion(blog, task(required_source_types=[]))["source_types"] is True
    assert completion(run(), task(min_supported_claims=5))["citation_min"] is False


def test_expected_tools_come_from_the_workflow_template():
    assert {"web_search", "github_search", "fetch_url"} <= expected_tools("technology_comparison")
    assert "arxiv_search" in expected_tools("literature_review") and "arxiv_search" not in expected_tools("fact_lookup")


def test_score_run_without_judge_covers_routing_tools_recovery_cost_and_latency():
    s = score_run(run(), task(), None, CFG)
    assert s["workflow_correct"] is True and s["tool_selection"] == 1.0 and s["plan_success"] is True
    assert s["cost_usd"] == pytest.approx(1.0 * 1 + 2.0 * 0.5) and s["llm_calls"] == 5 and s["routing_latency_ms"] == 10.0
    assert s["has_recovery"] is False and s["recovery_success"] is None and "citation_correctness" not in s
    wrong = score_run(run(workflow="market_research", router_intent="MARKET_RESEARCH"), task(), None, CFG)
    assert wrong["workflow_correct"] is False
    off = score_run(run(tools={"calls": 4, "searches": 2, "by_name": {"arxiv_search": 4}}), task(), None, CFG)
    assert off["tool_selection"] == 0.0
    rec = score_run(run(recoveries=[{"step": "a", "action": "retry"}]), task(), None, CFG)
    assert rec["has_recovery"] is True and rec["recovery_success"] is True


def test_routing_latency_adds_the_recorded_llm_call_when_the_supervisor_asked_the_llm():
    s = score_run(run(routing={"classifier_ms": 10.0, "llm_calls": 1, "llm_latency_ms": 900.0}, routing_llm_latency_ms=900.0), task(), None, CFG)
    assert s["routing_latency_ms"] == 910.0
    a = score_run(run(system="A", routing={"classifier_ms": 1.0, "llm_calls": 1, "llm_latency_ms": 800.0}, routing_llm_latency_ms=800.0), task(), None, CFG)
    assert a["routing_latency_ms"] == 801.0


def test_judged_metrics_unsupported_rate_citation_correctness_relevance_and_e2e():
    judged = {"claims": {"c0": "SUPPORTED", "c1": "SUPPORTED", "c2": "UNSUPPORTED"}, "relevance": {"s1": 2}}
    s = score_run(run(), task(), judged, CFG)
    assert s["unsupported_rate"] == pytest.approx(1 / 3) and s["citation_correctness"] == pytest.approx(2 / 3)
    assert s["source_relevance"] == 1.0 and s["e2e_success"] is False           # 0.67 < 0.8 and 0.33 > 0.2
    good = score_run(run(), task(), {"claims": {"c0": "SUPPORTED", "c1": "SUPPORTED", "c2": "SUPPORTED"}, "relevance": {"s1": 1}}, CFG)
    assert good["citation_correctness"] == 1.0 and good["source_relevance"] == 0.5 and good["e2e_success"] is True


def test_a_non_verbatim_quote_counts_against_citation_correctness_even_if_the_judge_agrees():
    r = run()
    r["evidence_checks"][0]["verbatim"] = False
    s = score_run(r, task(), {"claims": {"c0": "SUPPORTED", "c1": "SUPPORTED", "c2": "SUPPORTED"}, "relevance": {"s1": 2}}, CFG)
    assert s["citation_correctness"] == pytest.approx(2 / 3)


def test_failed_runs_without_evidence_score_cleanly():
    r = run(status="FAILED", report={"executive_summary": "failed", "findings": [], "evidence": []}, evidence_checks=[], cited_sources=[])
    s = score_run(r, task(), {"claims": {}, "relevance": {}}, CFG)
    assert s["completion"] is False and s["e2e_success"] is False and s["unsupported_rate"] is None and s["citation_correctness"] is None


def test_bootstrap_ci_is_seeded_bounds_the_mean_and_handles_edge_cases():
    xs = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(xs, 200, 0.95, seed=3)
    assert lo <= sum(xs) / len(xs) <= hi and (lo, hi) == bootstrap_ci(xs, 200, 0.95, seed=3)
    assert bootstrap_ci([], 100, 0.95, seed=1) == (None, None) and bootstrap_ci([1, 1], 100, 0.95, seed=1) == (1.0, 1.0)


def test_aggregate_computes_rates_percentiles_router_f1_and_ignores_missing_metrics():
    rows = [score_run(run(task_id=f"t{i}", wall_ms=100.0 * (i + 1)), task(id=f"t{i}"), None, CFG) for i in range(4)]
    rows[3]["workflow_correct"] = False
    agg = aggregate(rows, CFG, intents_true=["TECHNOLOGY_COMPARISON"] * 4,
                    intents_pred=["TECHNOLOGY_COMPARISON"] * 3 + ["MARKET_RESEARCH"])
    assert agg["n"] == 4 and agg["routing"]["accuracy"] == 0.75 and 0 < agg["routing"]["macro_f1"] < 1
    assert agg["wall_ms"]["p50"] == pytest.approx(250.0) and agg["efficiency"]["llm_calls"]["mean"] == 5.0
    assert agg["research"]["completion"]["mean"] == 1.0 and "ci95" in agg["research"]["completion"]
    assert "citation_correctness" not in agg["research"]            # judge not run
