import json

import pytest
from pydantic import ValidationError

from research_os.agents import build_agents
from research_os.agents.supervisor import Supervisor
from research_os.artifacts.render_markdown import render_markdown
from research_os.artifacts.render_pdf import PDFUnavailable, render_pdf
from research_os.artifacts.report import (
    Evidence,
    FindingOut,
    ResearchReport,
    SourceRef,
    build_report,
    generate_limitations,
)
from research_os.contracts import RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.demo_client import DemoLLM
from research_os.tools import build_registry


class Router:
    def __init__(self, workflow="technology_comparison", intent="TECHNOLOGY_COMPARISON", techs=("LangGraph", "CrewAI")):
        self.a = (workflow, intent, list(techs))

    def route(self, text):
        w, i, t = self.a
        return RouterOutput(intent=i, confidence=0.95, routing_mode=RoutingMode.DIRECT, workflow=w, latency_ms=1,
                            entities={"technologies": t, "companies": [], "topics": [], "time_scope": []})


def run(router=None, text="Compare LangGraph and CrewAI", llm=None):
    ctx = RunContext.create("rep1", {}, llm=llm or DemoLLM(), tools=build_registry(mode="fixture"))
    state = build_graph(router or Router(), Supervisor(), ctx, agents=build_agents()).invoke(initial_state(text))
    return state, ctx


@pytest.fixture(scope="module")
def comparison():
    state, ctx = run()
    return state, ctx, build_report(state["request"], state, ctx, llm_backend="demo")


# ---------- integrity ----------
def ev(cid="c1", sid="s1"):
    return Evidence(claim_id=cid, claim="A claim", source_id=sid, evidence_quote="a quote", verdict="SUPPORTED")


def base(**kw):
    d = dict(question="q", status="SUCCESS", executive_summary="s", methodology={}, findings=[FindingOut(statement="f", claim_ids=["c1"])],
             evidence=[ev()], sources=[SourceRef(source_id="s1", title="t", url="https://a.com", source_type="other")], run_metadata={})
    d.update(kw)
    return d


def test_valid_report_constructs():
    ResearchReport(**base())


def test_finding_without_a_claim_id_is_rejected():
    with pytest.raises(ValidationError, match="claim"):
        ResearchReport(**base(findings=[FindingOut(statement="uncited", claim_ids=[])]))


def test_finding_citing_an_unknown_claim_is_rejected():
    with pytest.raises(ValidationError, match="unknown claim"):
        ResearchReport(**base(findings=[FindingOut(statement="f", claim_ids=["ghost"])]))


def test_evidence_whose_source_is_missing_is_rejected():
    with pytest.raises(ValidationError, match="source"):
        ResearchReport(**base(evidence=[ev(sid="missing")]))


def test_only_supported_claims_may_back_findings():
    with pytest.raises(ValidationError, match="SUPPORTED"):
        ResearchReport(**base(evidence=[Evidence(claim_id="c1", claim="x", source_id="s1", evidence_quote="q", verdict="UNSUPPORTED")]))


# ---------- built from a real run ----------
def test_report_from_a_real_comparison_run_is_cited_and_consistent(comparison):
    state, ctx, rep = comparison
    assert rep.status == "SUCCESS" and rep.question == "Compare LangGraph and CrewAI" and rep.findings
    claim_ids = {e.claim_id for e in rep.evidence}
    assert all(c in claim_ids for f in rep.findings for c in f.claim_ids)
    src_ids = {s.source_id for s in rep.sources}
    for e in rep.evidence:                                                   # hard rule 8: verbatim quote in the stored source
        assert e.source_id in src_ids and e.evidence_quote in ctx.sources.get(e.source_id).content
    assert rep.methodology["workflow"] == "technology_comparison" and rep.methodology["router_mode"] == "DIRECT"
    assert {"search_agent", "verifier"} <= set(rep.methodology["agents"]) and "web_search" in rep.methodology["tools_used"]
    assert rep.run_metadata["verification"]["supported"] == len(rep.evidence)


def test_comparison_groups_verified_claims_per_entity(comparison):
    _, _, rep = comparison
    assert rep.comparison is not None
    assert [r["entity"] for r in rep.comparison] == ["LangGraph", "CrewAI"]
    assert all(r["claims"] for r in rep.comparison) and all({"claim_id", "statement"} <= set(c) for r in rep.comparison for c in r["claims"])


def test_non_comparison_workflow_has_no_comparison_table():
    state, ctx = run(Router("general_research", "GENERAL_RESEARCH", ()), "tell me about raft consensus leader election")
    assert build_report("q", state, ctx).comparison is None


def test_limitations_come_from_run_data_not_from_the_llm(comparison):
    _, _, rep = comparison
    text = " ".join(rep.limitations).lower()
    assert "demo" in text                                   # the run used the offline demo LLM
    assert "automated" in text                              # verification is automated
    assert not any("as an ai" in lim.lower() for lim in rep.limitations)


def test_limitations_mention_partial_budget_unsupported_and_conflicts():
    lim = generate_limitations({"status": "PARTIAL", "halted_reason": "budget exhausted: llm_calls limit 4 reached", "stats": {
        "unsupported": 3, "contradicted": 2, "deterministic_rejections": 2}, "conflicts": 1, "rejected_quotes": 4, "undated_sources": 2,
        "failed_steps": ["read:crewai"], "recoveries": 3, "llm_backend": "gemini", "flagged_sources": 1})
    t = " | ".join(lim).lower()
    for needle in ("partial", "budget", "3 claim", "conflict", "4 proposed", "2 source", "read:crewai", "3 recovery", "injection"):
        assert needle in t, needle


def test_partial_run_still_yields_a_valid_report():
    class Eat:
        def run(self, step, inputs, ctx, request):
            ctx.budget.count("llm_calls", 999)
    ctx = RunContext.create("p", {"budget": {"max_steps": 20, "max_tool_calls": 30, "max_searches": 10, "max_llm_calls": 1, "max_runtime_seconds": 60}},
                            llm=DemoLLM(), tools=build_registry(mode="fixture"))
    agents = {**build_agents(), "evidence_agent": Eat()}
    state = build_graph(Router(), Supervisor(), ctx, agents=agents).invoke(initial_state("Compare LangGraph and CrewAI"))
    rep = build_report("Compare LangGraph and CrewAI", state, ctx)
    assert rep.status == "PARTIAL" and rep.findings == [] and rep.executive_summary
    assert any("partial" in lim.lower() for lim in rep.limitations)


# ---------- renderers ----------
def test_markdown_has_every_section_numbered_citations_and_no_unsupported_text(comparison):
    _, _, rep = comparison
    md = render_markdown(rep)
    for heading in ("# ", "## Executive summary", "## Methodology", "## Findings", "## Comparison", "## Evidence", "## Limitations", "## Sources"):
        assert heading in md, heading
    assert "[1]" in md and md.index("## Findings") < md.index("## Evidence")
    first = rep.evidence[0]
    assert first.evidence_quote in md and first.claim in md
    assert "UNSUPPORTED" not in md.split("## Evidence")[1].split("## Limitations")[0]


def test_markdown_contradictions_show_both_sides():
    rep = ResearchReport(**base(contradictions=[{"subject": "LangGraph / persis", "sides": [
        {"stance": "asserts", "claims": [{"claim_id": "a", "claim": "LangGraph supports persistence", "source_id": "s1",
                                          "source_url": "https://a.com", "evidence_quote": "supports persistence"}]},
        {"stance": "negates", "claims": [{"claim_id": "b", "claim": "LangGraph has no persistence", "source_id": "s1",
                                          "source_url": "https://b.com", "evidence_quote": "has no persistence"}]}]}]))
    md = render_markdown(rep)
    assert "## Contradictions" in md and "supports persistence" in md and "has no persistence" in md and "asserts" in md and "negates" in md


def test_json_is_canonical_and_round_trips(comparison):
    _, _, rep = comparison
    again = ResearchReport.model_validate(json.loads(rep.model_dump_json()))
    assert again == rep


def test_pdf_renders_valid_pdf_bytes_with_the_report_text(comparison):
    _, _, rep = comparison
    try:
        pdf = render_pdf(rep)
    except PDFUnavailable:
        pytest.skip("WeasyPrint system libraries are not available")
    assert pdf[:5] == b"%PDF-" and pdf.rstrip().endswith(b"%%EOF") and len(pdf) > 3000
    from io import BytesIO

    from pypdf import PdfReader
    text = " ".join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert "Compare LangGraph and CrewAI" in text and "Limitations" in text


def test_renderers_refuse_a_report_that_violates_citation_integrity(comparison):
    _, _, rep = comparison
    broken = rep.model_copy(deep=True)
    broken.findings[0].claim_ids = []
    with pytest.raises(ValueError, match="citation"):
        render_markdown(broken)
    broken2 = rep.model_copy(deep=True)
    broken2.evidence[0].source_id = "nope"
    with pytest.raises(ValueError, match="citation"):
        render_markdown(broken2)
    with pytest.raises(ValueError, match="citation"):
        render_pdf(broken2)


def test_recall_and_failed_runs_produce_valid_reports():
    class S:
        pass
    ctx = RunContext.create("x", {}, llm=DemoLLM(), tools=build_registry(mode="fixture"))
    recall = {"request": "what did we find", "status": "SUCCESS", "recall": [{"run_id": "r1", "text": "old q", "similarity": 0.9,
              "report": {"executive_summary": "Earlier summary.", "findings": []}}]}
    rep = build_report("what did we find", recall, ctx)
    assert rep.status == "SUCCESS" and "Earlier summary." in rep.executive_summary and rep.run_metadata["recalled_from"] == ["r1"]
    failed = build_report("q", {}, ctx, status="FAILED", error="router exploded")
    assert failed.status == "FAILED" and "router exploded" in failed.executive_summary


def test_comparison_attributes_claims_to_the_entities_they_mention(comparison):
    _, _, rep = comparison
    rows = {r["entity"]: [c["statement"] for c in r["claims"]] for r in rep.comparison}
    assert any("crewai" in s.lower() for s in rows["CrewAI"])
    assert not any("crewai" in s.lower() and "langgraph" not in s.lower() for s in rows["LangGraph"])   # no CrewAI-only claim under LangGraph
    assert not any("langgraph" in s.lower() and "crewai" not in s.lower() for s in rows["CrewAI"])
