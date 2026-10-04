from evaluation.failure_injection import SCENARIOS, evaluate, render_md, run_scenario, run_suite


def test_there_are_at_least_the_ten_required_scenario_kinds():
    names = {s.name for s in SCENARIOS}
    assert len(SCENARIOS) >= 10 and {"search_api_timeout", "invalid_urls", "no_sources_found", "contradictory_sources", "malformed_tool_arguments",
                                     "llm_unsupported_claim", "tool_returns_garbage", "impossible_plan", "budget_exceeded",
                                     "prompt_injection_in_source"} <= names
    assert all(s.expect.get("statuses") for s in SCENARIOS)


def test_every_scenario_meets_its_declared_expectation_on_the_full_harness_and_never_crashes():
    bad = []
    for sc in SCENARIOS:
        r = run_scenario(sc, "D")
        assert r["crashed"] is False, f"{sc.name} crashed: {r.get('error')}"
        if not r["met"]:
            bad.append((sc.name, r["unmet"], r["status"], r["actions"]))
    assert bad == [], bad


def test_the_minimal_harness_is_visibly_worse_on_the_verification_scenarios():
    by = {s.name: s for s in SCENARIOS}
    d = run_scenario(by["llm_unsupported_claim"], "D")
    c = run_scenario(by["llm_unsupported_claim"], "C")
    assert d["met"] is True and c["met"] is False and any("always faster" in u for u in c["unmet"])
    d2, c2 = run_scenario(by["contradictory_sources"], "D"), run_scenario(by["contradictory_sources"], "C")
    assert d2["n_conflicts"] >= 1 and c2["n_conflicts"] == 0 and c2["met"] is False
    assert run_scenario(by["impossible_plan"], "C")["met"] is False       # no recovery: no replan


def test_suite_headline_rates_and_markdown():
    res = run_suite()
    d = res["summary"]["systems"]["D"]
    assert d["crash_rate"] == 0.0 and d["expectations_met"] == 1.0 and d["recovery_success_rate"] == 1.0 and d["safe_termination_rate"] == 1.0
    assert res["summary"]["systems"]["C"]["crash_rate"] == 0.0 and res["summary"]["systems"]["C"]["expectations_met"] < 1.0
    md = render_md(res)
    assert "| search_api_timeout |" in md and "NOT met" in md


def test_evaluate_reports_each_kind_of_mismatch():
    out = {"status": "FAILED", "actions": [], "n_conflicts": 0, "partial_message": None, "report_text": "x always faster", "evidence_text": "", "prompts_text": "",
           "event_text": ""}
    ok, why = evaluate(out, {"statuses": ["SUCCESS"], "actions_include": ["retry"], "partial_artifact": True, "min_conflicts": 1, "forbid_in_report": ["always faster"]})
    assert not ok and len(why) == 5
