import pytest

from research_os.contracts import FailureType
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.fixtures import FixtureCorpus
from research_os.tools.registry import ToolError


@pytest.fixture(scope="module")
def corpus():
    return FixtureCorpus.load("tests/fixtures/corpus")


def test_search_ranks_relevant_documents_first_and_is_deterministic(corpus):
    a = corpus.search("langgraph crewai comparison", "web_search", 5)
    assert a == corpus.search("langgraph crewai comparison", "web_search", 5)
    assert "blog.example-dev.io" in a[0].url
    assert all(s.tool == "web_search" for s in a)
    assert not any("weather" in s.url for s in a)


def test_search_respects_tool_and_limit(corpus):
    repos = corpus.search("langgraph crewai agents", "github_search", 5)
    assert repos and all(s.source_type == "repository" for s in repos)
    assert len(corpus.search("agents", "web_search", 2)) == 2
    assert corpus.search("zzzzqqq nothing", "web_search", 5) == []


def test_fetch_known_and_unknown_url(corpus):
    s = corpus.fetch("https://docs.example-agents.dev/langgraph/overview?utm_source=x")
    assert "stateful" in s.content and s.tool == "fetch_url"
    with pytest.raises(ToolError) as e:
        corpus.fetch("https://not-in-fixtures.example/x")
    assert e.value.failure_type == FailureType.INVALID_SOURCE


def make_ctx(reg, tmp_path, mode="off", rec=None):
    return RunContext.create("r", {}, llm=ScriptedLLM([]), tools=reg, runs_dir=str(tmp_path), mode=mode, recording_path=rec)


def test_fixture_registry_registers_all_five_tools_and_works(tmp_path):
    reg = build_registry(mode="fixture")
    assert reg.names() == ["arxiv_search", "fetch_url", "github_search", "pubmed_search", "web_search"]
    ctx = make_ctx(reg, tmp_path)
    assert ctx.tools.call("arxiv_search", {"query": "speculative decoding"}, ctx).sources[0].source_type == "paper"
    assert ctx.tools.call("pubmed_search", {"query": "migraine"}, ctx).sources
    assert ctx.tools.call("fetch_url", {"url": "https://docs.example-db.dev/raft"}, ctx).sources[0].title.startswith("Raft")
    assert ctx.budget.usage()["searches"] == 2  # arxiv + pubmed; fetch is not a search


def test_search_tools_count_against_search_budget_and_fetch_does_not(tmp_path):
    ctx = make_ctx(build_registry(mode="fixture"), tmp_path)
    ctx.tools.call("web_search", {"query": "langgraph"}, ctx)
    ctx.tools.call("fetch_url", {"url": "https://docs.example-db.dev/raft"}, ctx)
    assert ctx.budget.usage()["searches"] == 1 and ctx.budget.usage()["tool_calls"] == 2


def test_recorded_run_replays_offline_with_identical_outputs(tmp_path):
    rec = str(tmp_path / "rec.jsonl")
    live = make_ctx(build_registry(mode="fixture"), tmp_path, "record", rec)
    first = [live.tools.call("web_search", {"query": "vector database market"}, live),
             live.tools.call("arxiv_search", {"query": "small speech models"}, live)]
    with pytest.raises(ToolError):
        live.tools.call("fetch_url", {"url": "https://nope.example/x"}, live)
    live.recorder.save()

    exploding = build_registry(mode="fixture")     # same specs/schemas as the recording run

    def boom(args, ctx):
        raise AssertionError("tool must not execute during replay")
    for n in ("web_search", "arxiv_search", "fetch_url"):
        exploding._tools[n].fn = boom
    rep = make_ctx(exploding, tmp_path, "replay", rec)
    second = [rep.tools.call("web_search", {"query": "vector database market"}, rep),
              rep.tools.call("arxiv_search", {"query": "small speech models"}, rep)]
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]
    with pytest.raises(ToolError) as e:
        rep.tools.call("fetch_url", {"url": "https://nope.example/x"}, rep)
    assert e.value.failure_type == FailureType.INVALID_SOURCE


def test_live_mode_registers_real_tools_without_calling_them():
    assert build_registry(mode="live").names() == ["arxiv_search", "fetch_url", "github_search", "pubmed_search", "web_search"]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_registry(mode="warp")
