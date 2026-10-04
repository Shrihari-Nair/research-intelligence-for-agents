import json

import pytest

from research_os.contracts import FailureType, Source, source_id_for_url
from research_os.harness.context import RunContext
from research_os.harness.faults import FaultyLLM, FaultyRegistry, garbage_source
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.registry import ToolError, ToolResult


def ctx_with(reg, llm=None):
    return RunContext.create("f", {}, llm=llm or ScriptedLLM([]), tools=reg)


def test_fail_first_n_calls_then_recover_and_other_tools_are_untouched():
    reg = FaultyRegistry(build_registry(mode="fixture"))
    reg.fail("web_search", ToolError(FailureType.SEARCH_TIMEOUT, "injected"), first_n=2)
    c = ctx_with(reg)
    for _ in range(2):
        with pytest.raises(ToolError) as e:
            c.tools.call("web_search", {"query": "langgraph", "max_retries": 0} if False else {"query": "langgraph"}, c, agent="search_agent")
        assert e.value.failure_type == FailureType.SEARCH_TIMEOUT
    assert c.tools.call("web_search", {"query": "langgraph"}, c, agent="search_agent").sources        # third call passes through
    assert c.tools.call("github_search", {"query": "langgraph"}, c, agent="search_agent").sources    # untouched
    assert reg.injected["web_search"] == 2


def test_empty_results_and_replaced_results():
    reg = FaultyRegistry(build_registry(mode="fixture"))
    reg.empty("web_search")
    c = ctx_with(reg)
    assert c.tools.call("web_search", {"query": "langgraph"}, c, agent="search_agent").sources == []
    reg2 = FaultyRegistry(build_registry(mode="fixture"))
    bad = garbage_source("https://junk.example/x")
    reg2.replace("web_search", ToolResult(sources=[bad]))
    c2 = ctx_with(reg2)
    assert c2.tools.call("web_search", {"query": "x"}, c2, agent="search_agent").sources[0].url == "https://junk.example/x"


def test_tamper_arguments_triggers_schema_validation_failure_once():
    reg = FaultyRegistry(build_registry(mode="fixture"))
    reg.tamper_args("web_search", {"query": ""}, first_n=1)
    c = ctx_with(reg)
    with pytest.raises(ToolError) as e:
        c.tools.call("web_search", {"query": "langgraph"}, c, agent="search_agent")
    assert "invalid arguments" in str(e.value)
    assert c.tools.call("web_search", {"query": "langgraph"}, c, agent="search_agent").sources


def test_proxy_keeps_the_registry_interface():
    reg = FaultyRegistry(build_registry(mode="fixture"))
    assert reg.names() == ["arxiv_search", "fetch_url", "github_search", "pubmed_search", "web_search"]


def test_garbage_source_is_a_valid_source_with_unusable_text():
    s = garbage_source("https://junk.example/x")
    assert isinstance(s, Source) and s.source_id == source_id_for_url("https://junk.example/x") and len(s.content.split(".")) <= 2


def test_llm_fault_malformed_first_n_calls_for_a_given_prompt():
    base = ScriptedLLM(["good1", "good2"])
    llm = FaultyLLM(base)
    llm.malformed("evidence", first_n=2)
    assert llm.complete("x", meta={"prompt": "evidence.v2"}).text != "good1"
    assert llm.complete("x", meta={"prompt": "evidence.v2"}).text != "good1"
    assert llm.complete("x", meta={"prompt": "evidence.v2"}).text == "good1"          # third call reaches the base model
    assert llm.complete("x", meta={"prompt": "verifier_judge.v1"}).text == "good2"    # other prompts untouched
    assert llm.injected == 2


def test_llm_fault_rewrites_evidence_output_to_add_an_overreaching_claim():
    class Evid(ScriptedLLM):
        pass
    prompt = "Passages:\n<<<SOURCE id=s1\nLangGraph supports persistence through checkpointers.\nSOURCE>>>"
    llm = FaultyLLM(Evid([json.dumps({"claims": []})]))
    llm.add_overreaching_claim("LangGraph is always faster than CrewAI")
    out = json.loads(llm.complete(prompt, meta={"prompt": "evidence.v2"}).text)
    assert out["claims"][0]["claim"] == "LangGraph is always faster than CrewAI" and out["claims"][0]["source_id"] == "s1"
    assert out["claims"][0]["evidence_quote"] in prompt
