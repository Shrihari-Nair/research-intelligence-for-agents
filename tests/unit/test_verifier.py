import json
from datetime import datetime, timezone

import pytest

from research_os.agents.verifier import Verifier
from research_os.contracts import ResearchStep, Source, source_id_for_url
from research_os.graph.templates import get_workflow
from research_os.harness.context import RunContext
from research_os.harness.verification import detect_conflicts, deterministic_checks, gate
from research_os.llm.mock_client import ScriptedLLM
from research_os.llm.structured import MalformedLLMOutput
from research_os.tools import build_registry

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def src(url, content, typ="documentation", published=NOW):
    return Source(source_id=source_id_for_url(url), title="t", url=url, source_type=typ, published_at=published,
                  retrieved_at=NOW, content=content, tool="web_search")


A = src("https://docs.x.dev/a", "LangGraph supports persistence through checkpointers. It is released under the MIT license.")
B = src("https://blog.x.dev/b", "LangGraph is a library for agents. We did not run any latency benchmarks.", "blog")
OLD = src("https://old.x.dev/c", "In this early release LangGraph has no persistence: graph state lives only in memory.", "blog")


def claim(cid, text, source, quote, verdict="UNVERIFIED"):
    return {"claim_id": cid, "claim": text, "source_id": source.source_id, "evidence_quote": quote, "extractor_confidence": 0.9,
            "verdict": verdict, "verdict_rationale": None}


def ctx_with(sources, responses=()):
    c = RunContext.create("r", {}, llm=ScriptedLLM(list(responses)), tools=build_registry(mode="fixture"))
    for s in sources:
        c.sources.add(s)
    return c


def judge(*items):
    return json.dumps({"judgements": [{"claim_id": i, "verdict": v, "rationale": r} for i, v, r in items]})


WF = get_workflow("general_research")


# ---------- layer 1: deterministic ----------
def reasons(c, sources, wf=WF):
    return deterministic_checks(c, {s.source_id: s for s in sources}, wf)


def test_valid_claim_passes_deterministic_layer():
    assert reasons(claim("1", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers"), [A]) == []


def test_missing_source_and_fabricated_quote_fail():
    c = claim("1", "x", A, "LangGraph supports persistence")
    c["source_id"] = "nope"
    assert any("source" in r for r in reasons(c, [A]))
    assert any("verbatim" in r for r in reasons(claim("2", "LangGraph is 100x faster", A, "LangGraph is 100 times faster"), [A]))


def test_unsupported_comparative_without_comparative_quote_fails():
    r = reasons(claim("1", "LangGraph is faster than CrewAI", B, "LangGraph is a library for agents"), [B])
    assert any("comparative" in x for x in r)


def test_comparative_with_matching_quote_is_left_to_the_judge():
    s = src("https://docs.x.dev/p", "In our test LangGraph was faster than CrewAI by 20 percent.")
    assert reasons(claim("1", "LangGraph is faster than CrewAI", s, "LangGraph was faster than CrewAI by 20 percent"), [s]) == []


def test_numbers_universals_and_causal_language_must_appear_in_the_quote():
    assert any("number" in r for r in reasons(claim("1", "LangGraph has 500 contributors", A, "LangGraph supports persistence"), [A]))
    assert any("universal" in r for r in reasons(claim("2", "LangGraph always persists state", A, "LangGraph supports persistence"), [A]))
    c3 = claim("3", "Checkpointers cause better recovery", A, "LangGraph supports persistence through checkpointers")
    assert any("causal" in r for r in reasons(c3, [A]))


def test_source_type_and_date_rules():
    wf = {**WF, "allowed_source_types": ["documentation"]}
    assert any("source type" in r for r in reasons(claim("1", "LangGraph is a library", B, "LangGraph is a library for agents"), [B], wf))
    undated = src("https://x.dev/u", "Market size was 1.5 billion in 2024.", "news", None)
    mw = get_workflow("market_research")
    assert any("date" in r for r in reasons(claim("1", "size", undated, "Market size was 1.5 billion in 2024"), [undated], mw))
    assert reasons(claim("1", "size", src("https://x.dev/d", "Market size was 1.5 billion in 2024.", "news"), "Market size was 1.5 billion in 2024"),
                   [src("https://x.dev/d", "Market size was 1.5 billion in 2024.", "news")], mw) == []


# ---------- conflicts ----------
def test_opposite_statements_from_two_sources_produce_a_conflict():
    cs = [claim("a", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers", "SUPPORTED"),
          claim("b", "LangGraph has no persistence", OLD, "LangGraph has no persistence", "SUPPORTED")]
    conflicts = detect_conflicts(cs, {s.source_id: s for s in (A, OLD)})
    assert len(conflicts) == 1 and {x["claim_id"] for side in conflicts[0]["sides"] for x in side["claims"]} == {"a", "b"}
    assert "LangGraph" in conflicts[0]["subject"] and "persis" in conflicts[0]["subject"]


def test_agreeing_or_unrelated_claims_do_not_conflict():
    cs = [claim("a", "LangGraph supports persistence", A, "q", "SUPPORTED"),
          claim("b", "LangGraph is released under the MIT license", A, "q", "SUPPORTED")]
    assert detect_conflicts(cs, {A.source_id: A}) == []


# ---------- gate ----------
def test_gate_pass_and_fail_paths():
    wf = {**WF, "min_supported_claims": 2, "max_unsupported_rate": 0.5, "required_source_types": []}
    ok = gate({"SUPPORTED": 2, "CONTRADICTED": 0, "UNSUPPORTED": 1}, [], wf)
    assert ok["result"] == "PASS"
    few = gate({"SUPPORTED": 1, "CONTRADICTED": 0, "UNSUPPORTED": 0}, [], wf)
    assert few["result"] == "FAIL" and "min_supported_claims" in few["reason"]
    noisy = gate({"SUPPORTED": 2, "CONTRADICTED": 0, "UNSUPPORTED": 5}, [], wf)
    assert noisy["result"] == "FAIL" and "unsupported" in noisy["reason"]
    assert gate({"SUPPORTED": 0, "CONTRADICTED": 0, "UNSUPPORTED": 0}, [], wf)["result"] == "FAIL"


def test_gate_requires_a_supported_claim_from_a_required_source_type():
    wf = {**WF, "min_supported_claims": 1, "max_unsupported_rate": 1.0, "required_source_types": ["documentation"]}
    assert gate({"SUPPORTED": 1, "CONTRADICTED": 0, "UNSUPPORTED": 0}, ["blog"], wf)["result"] == "FAIL"
    assert gate({"SUPPORTED": 1, "CONTRADICTED": 0, "UNSUPPORTED": 0}, ["documentation"], wf)["result"] == "PASS"


# ---------- layer 2 + orchestration ----------
def step(workflow="general_research"):
    return ResearchStep(id="verify", objective="verify", agent="verifier", tools=[], depends_on=["extract"],
                        params={"workflow": workflow})


def run(claims, sources, responses, workflow="general_research"):
    c = ctx_with(sources, responses)
    return Verifier().run(step(workflow), {"extract": {"claims": claims}}, c, "r"), c


def test_unsupported_comparative_is_never_supported_and_never_reaches_the_judge():
    bad = claim("c1", "LangGraph is faster than CrewAI", B, "LangGraph is a library for agents")
    out, c = run([bad], [B], [])
    assert out["claims"][0]["verdict"] == "UNSUPPORTED" and "comparative" in out["claims"][0]["verdict_rationale"]
    assert c.llm.base.calls == 0 and out["gate"]["result"] == "FAIL"


def test_judge_verdicts_are_applied_and_counted():
    cs = [claim("c1", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers"),
          claim("c2", "LangGraph is MIT licensed", A, "It is released under the MIT license"),
          claim("c3", "LangGraph is open source", A, "It is released under the MIT license")]
    out, c = run(cs, [A], [judge(("c1", "SUPPORTED", "direct"), ("c2", "SUPPORTED", "direct"), ("c3", "UNSUPPORTED", "MIT does not mean open source"))])
    v = {x["claim_id"]: x["verdict"] for x in out["claims"]}
    assert v == {"c1": "SUPPORTED", "c2": "SUPPORTED", "c3": "UNSUPPORTED"} and c.llm.base.calls == 1
    assert out["stats"]["supported"] == 2 and out["stats"]["unsupported"] == 1 and out["gate"]["result"] == "PASS"
    assert c.llm.base.metas[0] == {"prompt": "verifier_judge.v1"}


def test_judge_can_mark_contradicted():
    out, _ = run([claim("c1", "LangGraph has persistence", OLD, "LangGraph has no persistence")], [OLD],
                 [judge(("c1", "CONTRADICTED", "quote says the opposite"))])
    assert out["claims"][0]["verdict"] == "CONTRADICTED"


def test_judge_missing_a_claim_triggers_retry_and_garbage_raises():
    cs = [claim("c1", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers"),
          claim("c2", "LangGraph is MIT licensed", A, "It is released under the MIT license")]
    out, c = run(cs, [A], [judge(("c1", "SUPPORTED", "ok")), judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok"))])
    assert c.llm.base.calls == 2 and {x["verdict"] for x in out["claims"]} == {"SUPPORTED"}
    with pytest.raises(MalformedLLMOutput):
        run(cs, [A], ["junk"] * 3)


def test_judge_cannot_upgrade_a_deterministic_rejection():
    bad = claim("c1", "LangGraph is faster than CrewAI", B, "LangGraph is a library for agents")
    good = claim("c2", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers")
    out, c = run([bad, good], [A, B], [judge(("c1", "SUPPORTED", "ok"))])   # the judge sees ONLY the good claim, as short id c1
    assert {x["claim_id"]: x["verdict"] for x in out["claims"]} == {"c1": "UNSUPPORTED", "c2": "SUPPORTED"}
    assert bad["claim"] not in c.llm.base.prompts[0] and good["claim"] in c.llm.base.prompts[0]


def test_conflicting_supported_claims_become_contradicted_and_listed():
    cs = [claim("a", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers"),
          claim("b", "LangGraph has no persistence", OLD, "LangGraph has no persistence")]
    out, _ = run(cs, [A, OLD], [judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok")), confirm(("k1", True))])
    assert {x["verdict"] for x in out["claims"]} == {"CONTRADICTED"} and len(out["conflicts"]) == 1
    assert "conflict" in out["claims"][0]["verdict_rationale"].lower()


def test_judge_prompt_marks_quotes_as_untrusted_data_and_is_delimited():
    cs = [claim("c1", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers")]
    _, c = run(cs, [A], [judge(("c1", "SUPPORTED", "ok"))])
    p = c.llm.base.prompts[0]
    assert "<<<CLAIM id=c1" in p and "untrusted" in p.lower() and "CLAIM>>>" in p


def test_no_claims_gives_fail_gate_without_llm_calls():
    out, c = run([], [A], [])
    assert out["gate"]["result"] == "FAIL" and c.llm.base.calls == 0 and out["claims"] == []


def test_every_verdict_records_where_it_came_from():
    cs = [claim("c1", "LangGraph is faster than CrewAI", B, "LangGraph is a library for agents"),
          claim("c2", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers")]
    out, _ = run(cs, [A, B], [judge(("c1", "SUPPORTED", "ok"))])
    assert {c["claim_id"]: c["verdict_source"] for c in out["claims"]} == {"c1": "deterministic", "c2": "judge"}


def test_conflicting_numbers_go_to_the_judge_but_numbers_with_no_quote_numbers_are_rejected():
    s = src("https://x.dev/n", "This pilot trial enrolled 40 participants over six months.")
    c = claim("1", "The trial enrolled 400 participants", s, "This pilot trial enrolled 40 participants")
    assert reasons(c, [s]) == []                      # quote has a (different) number: the judge can say CONTRADICTED
    nonumber = claim("2", "The trial enrolled 400 participants", s, "This pilot trial enrolled")
    assert any("number" in r for r in reasons(nonumber, [s]))


# ---- conflicts are evidence, not absence of evidence ----
def test_gate_passes_when_the_only_claims_are_in_a_detected_conflict():
    wf = {**WF, "min_supported_claims": 2, "max_unsupported_rate": 0.5, "required_source_types": ["documentation"]}
    assert gate({"SUPPORTED": 0, "CONTRADICTED": 2, "UNSUPPORTED": 0}, [], wf)["result"] == "FAIL"             # no conflict info: unchanged
    g = gate({"SUPPORTED": 0, "CONTRADICTED": 2, "UNSUPPORTED": 0}, [], wf, n_conflicts=1)
    assert g["result"] == "PASS" and "conflict" in (g["note"] or "").lower()


def test_conflicts_do_not_excuse_a_high_unsupported_rate():
    wf = {**WF, "min_supported_claims": 2, "max_unsupported_rate": 0.3, "required_source_types": []}
    assert gate({"SUPPORTED": 0, "CONTRADICTED": 2, "UNSUPPORTED": 6}, [], wf, n_conflicts=1)["result"] == "FAIL"


def test_verifier_run_with_conflicting_sources_passes_its_gate_and_lists_the_conflict():
    cs = [claim("a", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers"),
          claim("b", "LangGraph has no persistence", OLD, "LangGraph has no persistence")]
    out, _ = run(cs, [A, OLD], [judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok")), confirm(("k1", True))])
    assert out["gate"]["result"] == "PASS" and len(out["conflicts"]) == 1 and out["stats"]["supported"] == 0


def test_gate_counts_only_conflict_claims_not_judge_contradictions():
    wf = {**WF, "min_supported_claims": 3, "max_unsupported_rate": 0.9, "required_source_types": []}
    counts = {"SUPPORTED": 0, "CONTRADICTED": 4, "UNSUPPORTED": 0}
    assert gate(counts, [], wf, n_conflicts=1, n_conflict_claims=2)["result"] == "FAIL"   # 2 conflict claims + 2 judge-contradicted
    assert gate(counts, [], wf, n_conflicts=1, n_conflict_claims=4)["result"] == "PASS"


# ---- conflict precision: different sources, then an LLM confirmation ----
def confirm(*items):
    return json.dumps({"checks": [{"id": i, "contradict": v, "reason": "r"} for i, v in items]})


PAIR = [claim("a", "LangGraph supports persistence", A, "LangGraph supports persistence through checkpointers"),
        claim("b", "LangGraph has no persistence", OLD, "LangGraph has no persistence")]


def test_opposite_claims_from_the_same_source_are_not_a_conflict():
    s = src("https://one.x.dev/p", "LangGraph supports persistence in v1. LangGraph has no persistence in the free tier.")
    cs = [claim("a", "LangGraph supports persistence", s, "LangGraph supports persistence in v1"),
          claim("b", "LangGraph has no persistence", s, "LangGraph has no persistence in the free tier")]
    assert detect_conflicts(cs, {s.source_id: s}) == []


def test_a_conflict_the_llm_does_not_confirm_leaves_the_claims_supported():
    out, c = run(PAIR, [A, OLD], [judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok")), confirm(("k1", False))])
    assert out["conflicts"] == [] and out["stats"]["supported"] == 2 and out["stats"]["conflict_candidates"] == 1
    assert "k1" in c.llm.base.prompts[1] and "persistence" in c.llm.base.prompts[1]


def test_a_confirmed_conflict_still_contradicts_both_sides():
    out, _ = run(PAIR, [A, OLD], [judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok")), confirm(("k1", True))])
    assert len(out["conflicts"]) == 1 and {x["verdict"] for x in out["claims"]} == {"CONTRADICTED"}


def test_a_failed_confirmation_is_treated_as_unconfirmed_not_as_a_conflict():
    bad = ["not json"] * 3
    out, _ = run(PAIR, [A, OLD], [judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok")), *bad])
    assert out["conflicts"] == [] and out["stats"]["supported"] == 2


def test_a_candidate_whose_sides_are_about_different_subjects_is_not_confirmed_even_if_marked_contradict():
    reply = json.dumps({"checks": [{"id": "k1", "subject_a": "LangGraph Server", "subject_b": "a Docker image", "same_subject": False,
                                    "contradict": True, "reason": "r"}]})
    out, _ = run(PAIR, [A, OLD], [judge(("c1", "SUPPORTED", "ok"), ("c2", "SUPPORTED", "ok")), reply])
    assert out["conflicts"] == [] and out["stats"]["supported"] == 2
