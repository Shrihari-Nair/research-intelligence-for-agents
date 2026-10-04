import json

import pytest

from research_os.agents import build_agents
from research_os.agents.base import AgentFailure
from research_os.agents.evidence_agent import EvidenceAgent
from research_os.agents.search_agent import SearchAgent
from research_os.agents.source_agent import SourceAgent
from research_os.agents.synthesis_agent import SynthesisAgent
from research_os.contracts import FailureType, ResearchStep, source_id_for_url
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.llm.structured import MalformedLLMOutput
from research_os.tools import build_registry
from research_os.tools.registry import ToolError, ToolRegistry, ToolResult

LG = "https://docs.example-agents.dev/langgraph/overview"
LG_ID = source_id_for_url(LG)
EVIL = "https://evil.example-site.net/langgraph-tips"


def ctx(responses=(), tools=None):
    return RunContext.create("r", {}, llm=ScriptedLLM(list(responses)), tools=tools or build_registry(mode="fixture"))


def step(id, agent, tools=(), deps=(), **params):
    return ResearchStep(id=id, objective=params.pop("objective", "Find sources for LangGraph"), agent=agent, tools=list(tools),
                        depends_on=list(deps), params={"request": "Compare LangGraph and CrewAI", **params})


# ---------------- SearchAgent ----------------
def test_search_agent_returns_ranked_deduped_source_ids_and_queries():
    c = ctx()
    out = SearchAgent().run(step("search:langgraph", "search_agent", ["web_search", "github_search"], entity="LangGraph"), {}, c, "r")
    assert LG_ID in out["source_ids"] and len(out["source_ids"]) == len(set(out["source_ids"]))
    assert any("LangGraph" in q["query"] for q in out["queries"]) and {q["tool"] for q in out["queries"]} == {"web_search", "github_search"}
    assert all(c.sources.get(i) is not None for i in out["source_ids"])
    assert c.budget.usage()["searches"] == 2


def test_search_agent_prefers_allowed_source_types():
    c = ctx()
    out = SearchAgent().run(step("s", "search_agent", ["web_search"], entity="LangGraph", allowed_source_types=["documentation"]), {}, c, "r")
    assert c.sources.get(out["source_ids"][0]).source_type == "documentation"


def test_search_agent_passes_recent_sort_to_scholarly_tools():
    seen = {}
    reg = ToolRegistry()
    reg.register("arxiv_search", lambda a, c: seen.update(a) or ToolResult(), cost_class="search")
    with pytest.raises(AgentFailure):
        SearchAgent().run(step("s", "search_agent", ["arxiv_search"], recent=True), {}, ctx(tools=reg), "r")
    assert seen["sort"] == "recent"


def test_search_agent_survives_one_failing_tool_but_raises_when_all_fail():
    reg = build_registry(mode="fixture")

    def boom(a, c):
        raise ToolError(FailureType.SEARCH_TIMEOUT, "slow")
    reg.register("github_search", boom, cost_class="search")
    out = SearchAgent().run(step("s", "search_agent", ["web_search", "github_search"], entity="LangGraph"), {}, ctx(tools=reg), "r")
    assert out["source_ids"] and out["errors"][0]["failure_type"] == "SEARCH_TIMEOUT"
    only_bad = ToolRegistry()
    only_bad.register("web_search", boom, cost_class="search")
    with pytest.raises(ToolError) as e:
        SearchAgent().run(step("s", "search_agent", ["web_search"]), {}, ctx(tools=only_bad), "r")
    assert e.value.failure_type == FailureType.SEARCH_TIMEOUT


def test_search_agent_no_results_is_no_useful_sources():
    with pytest.raises(AgentFailure) as e:
        SearchAgent().run(step("s", "search_agent", ["web_search"], entity="zzzqqq", request="zzzqqq"), {}, ctx(), "r")
    assert e.value.failure_type == FailureType.NO_USEFUL_SOURCES


# ---------------- SourceAgent ----------------
def search_out(c, *urls):
    from research_os.tools.fixtures import FixtureCorpus
    corpus = FixtureCorpus.load("tests/fixtures/corpus")
    for u in urls:
        c.sources.add(corpus.fetch(u))
    return {"search": {"source_ids": [source_id_for_url(u) for u in urls]}}


def test_source_agent_passages_keep_offsets_into_source_content():
    c = ctx()
    out = SourceAgent().run(step("read", "source_agent", ["fetch_url"], ["search"], objective="Extract passages about LangGraph persistence"),
                            search_out(c, LG), c, "r")
    assert out["passages"]
    for p in out["passages"]:
        assert c.sources.get(p["source_id"]).content[p["start"]:p["end"]] == p["text"]
    assert any("persistence" in p["text"] for p in out["passages"])


def test_source_agent_fetches_full_pages_via_registry_only_when_tools_allow():
    c = ctx()
    SourceAgent().run(step("read", "source_agent", ["fetch_url"], ["search"]), search_out(c, LG), c, "r")
    c2 = ctx()
    SourceAgent().run(step("read", "source_agent", [], ["search"]), search_out(c2, LG), c2, "r")
    assert c2.budget.usage()["tool_calls"] == 0


def test_prompt_injection_sentences_never_become_passages_and_are_flagged():
    c = ctx()
    out = SourceAgent().run(step("read", "source_agent", [], ["search"], objective="LangGraph tips"), search_out(c, EVIL), c, "r")
    joined = " ".join(p["text"] for p in out["passages"])
    assert "IGNORE ALL PREVIOUS" not in joined and "admin mode" not in joined
    assert "start with a simple graph" in joined               # the benign part of the page survives
    assert out["flagged_sources"][0]["source_id"] == source_id_for_url(EVIL)
    assert any(e.type == "info" and "injection" in json.dumps(e.payload) for e in c.events.events)


def test_source_agent_without_usable_text_is_no_useful_sources():
    c = ctx()
    with pytest.raises(AgentFailure) as e:
        SourceAgent().run(step("read", "source_agent", [], ["search"]), {"search": {"source_ids": []}}, c, "r")
    assert e.value.failure_type == FailureType.NO_USEFUL_SOURCES


# ---------------- EvidenceAgent ----------------
QUOTE = "It supports persistence through checkpointers"


def passages_for(c):
    out = SourceAgent().run(step("read", "source_agent", [], ["search"], objective="LangGraph persistence"), search_out(c, LG), c, "r")
    return {"read": out}


def claims_json(*items):
    return json.dumps({"claims": list(items)})


def item(claim="LangGraph supports persistence", source_id="s1", quote=QUOTE, confidence=0.9):   # s1 = the only passage source (short id)
    return {"claim": claim, "source_id": source_id, "evidence_quote": quote, "confidence": confidence}


def test_evidence_agent_keeps_valid_claims_with_verbatim_quotes():
    c = ctx([claims_json(item())])
    out = EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"], objective="claims about LangGraph"), passages_for(c), c, "r")
    assert len(out["claims"]) == 1 and out["rejected_quotes"] == 0
    cl = out["claims"][0]
    assert cl["source_id"] == LG_ID and cl["evidence_quote"] == QUOTE and cl["verdict"] == "UNVERIFIED"
    assert QUOTE in c.sources.get(LG_ID).content and len(cl["claim_id"]) == 12


def test_fabricated_quotes_and_unknown_sources_are_rejected_deterministically():
    c = ctx([claims_json(item(), item("fast", quote="LangGraph is 100 times faster than CrewAI"),
                         item("ghost", source_id="deadbeefdeadbeef"), item("dup"))])
    out = EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"]), passages_for(c), c, "r")
    assert [x["claim"] for x in out["claims"]] == ["LangGraph supports persistence"]  # duplicate quote also collapsed
    assert out["rejected_quotes"] == 2


def test_prompt_is_delimited_labels_content_untrusted_and_logs_prompt_version():
    c = ctx([claims_json(item())])
    EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"]), passages_for(c), c, "r")
    p = c.llm.base.prompts[0]
    assert "<<<SOURCE id=s1" in p and LG_ID not in p and "SOURCE>>>" in p and "untrusted" in p.lower()   # short ids, never the hash
    assert c.llm.base.metas[0] == {"prompt": "evidence.v2"}


def test_injection_text_never_reaches_the_evidence_prompt():
    c = ctx([claims_json()])
    out = SourceAgent().run(step("read", "source_agent", [], ["search"], objective="LangGraph tips"), search_out(c, EVIL), c, "r")
    EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"]), {"read": out}, c, "r")
    assert "IGNORE ALL PREVIOUS" not in c.llm.base.prompts[0] and "admin mode" not in c.llm.base.prompts[0]


def test_invalid_llm_output_raises_malformed_after_bounded_retries_and_bad_confidence_retries():
    c = ctx(["nope", "nope", "nope"])
    with pytest.raises(MalformedLLMOutput):
        EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"]), passages_for(c), c, "r")
    assert c.llm.base.calls == 3
    c2 = ctx([claims_json(item(confidence=7)), claims_json(item())])
    assert len(EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"]), passages_for(c2), c2, "r")["claims"]) == 1


def test_no_passages_is_no_useful_sources():
    with pytest.raises(AgentFailure) as e:
        EvidenceAgent().run(step("extract", "evidence_agent", [], ["read"]), {"read": {"passages": []}}, ctx(), "r")
    assert e.value.failure_type == FailureType.NO_USEFUL_SOURCES


# ---------------- SynthesisAgent ----------------
def claim(cid, verdict, text="LangGraph supports persistence"):
    return {"claim_id": cid, "claim": text, "source_id": LG_ID, "evidence_quote": QUOTE, "extractor_confidence": 0.9,
            "verdict": verdict, "verdict_rationale": None}


def synth_json(*findings, summary="Summary."):
    return json.dumps({"summary": summary, "findings": list(findings)})


def test_synthesis_uses_only_supported_claims_and_cites_claim_ids():
    c = ctx([synth_json({"statement": "LangGraph supports persistence.", "claim_ids": ["c1"]})])
    inputs = {"verify": {"claims": [claim("c1", "SUPPORTED"), claim("c2", "UNSUPPORTED", "faster"), claim("c3", "CONTRADICTED")]}}
    out = SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), inputs, c, "r")
    assert out["findings"] == [{"statement": "LangGraph supports persistence.", "claim_ids": ["c1"]}]
    p = c.llm.base.prompts[0]
    assert "c1" in p and "c2" not in p and "c3" not in p


def test_synthesis_rejects_unknown_claim_ids_by_retrying():
    c = ctx([synth_json({"statement": "x", "claim_ids": ["ghost"]}), synth_json({"statement": "ok", "claim_ids": ["c1"]})])
    out = SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), {"verify": {"claims": [claim("c1", "SUPPORTED")]}}, c, "r")
    assert out["findings"][0]["claim_ids"] == ["c1"] and c.llm.base.calls == 2


def test_synthesis_rejects_findings_without_citations():
    c = ctx([synth_json({"statement": "uncited", "claim_ids": []})] * 3)
    with pytest.raises(MalformedLLMOutput):
        SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), {"verify": {"claims": [claim("c1", "SUPPORTED")]}}, c, "r")


def test_synthesis_with_no_supported_claims_makes_no_llm_call():
    c = ctx([])
    out = SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), {"verify": {"claims": [claim("c2", "UNSUPPORTED")]}}, c, "r")
    assert out["findings"] == [] and "no verified" in out["summary"].lower() and c.llm.base.calls == 0


def test_build_agents_registers_the_named_agents():
    assert {"search_agent", "source_agent", "evidence_agent", "synthesis_agent"} <= set(build_agents())


def test_evidence_prompt_caps_the_number_of_claims_and_extra_claims_are_dropped():
    from research_os.agents.evidence_agent import EvidenceAgent
    c = ctx([claims_json(*[item(f"claim {i}", quote=QUOTE) for i in range(5)])])
    agent = EvidenceAgent({"evidence": {"batch_chars": 6000, "max_batches": 2, "max_quote_chars": 300, "max_claims_per_batch": 2,
                                        "max_output_tokens": 4096}})
    out = agent.run(step("extract", "evidence_agent", [], ["read"]), passages_for(c), c, "r")
    assert "at most 2 claims" in c.llm.base.prompts[0]
    assert len(out["claims_considered"]) <= 2 if "claims_considered" in out else True
    assert out["claims_proposed"] <= 2


# ---- numbers in free text must come from the cited claims ----
def num_claim(cid, text, quote):
    return {**claim(cid, "SUPPORTED", text), "evidence_quote": quote}


def test_ungrounded_numbers_in_a_finding_force_a_retry_with_a_hint():
    from research_os.harness.verification import ungrounded_numbers
    assert ungrounded_numbers("It has 12,000 stars and 3.5x speed", ["Stars: 12000"]) == ["3.5"]
    assert ungrounded_numbers("Nothing numeric here", ["Stars: 12000"]) == []
    c = ctx([synth_json({"statement": "LangGraph has 99 stars.", "claim_ids": ["c1"]}),
             synth_json({"statement": "LangGraph has 12000 stars.", "claim_ids": ["c1"]})])
    claims = [num_claim("c1", "LangGraph has 12000 stars", "Stars: 12000")]
    out = SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), {"verify": {"claims": claims}}, c, "r")
    assert out["findings"][0]["statement"] == "LangGraph has 12000 stars." and c.llm.base.calls == 2
    assert "99" in c.llm.base.prompts[1]                       # the retry names the offending number


def test_ungrounded_number_in_the_summary_is_rejected_but_a_number_from_the_request_is_allowed():
    claims = [num_claim("c1", "LangGraph has 12000 stars", "Stars: 12000")]
    bad = ctx([synth_json({"statement": "LangGraph has 12000 stars.", "claim_ids": ["c1"]}, summary="It grew 400% last year.")] * 3)
    with pytest.raises(MalformedLLMOutput):
        SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), {"verify": {"claims": claims}}, bad, "r")
    ok = ctx([synth_json({"statement": "LangGraph has 12000 stars.", "claim_ids": ["c1"]}, summary="The top 5 frameworks include LangGraph.")])
    out = SynthesisAgent().run(step("synthesize", "synthesis_agent", [], ["verify"]), {"verify": {"claims": claims}}, ok, "show the top 5 frameworks")
    assert out["summary"].startswith("The top 5")
