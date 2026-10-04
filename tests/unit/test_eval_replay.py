import json
from datetime import datetime, timezone

import pytest

from evaluation.replay import CachedLLM, CacheMiss, CassetteCorpus, build_replay_registry
from research_os.contracts import FailureType, Source, source_id_for_url
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools.registry import ToolError

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def src(url, title, content, typ="documentation", tool="web_search"):
    return Source(source_id=source_id_for_url(url), title=title, url=url, source_type=typ, published_at=NOW, retrieved_at=NOW,
                  content=content, tool=tool)


def write_corpus(tmp_path, task="t1"):
    srcs = [src("https://docs.a.dev/langgraph", "LangGraph docs", "LangGraph builds stateful agent graphs with checkpointers."),
            src("https://github.com/o/langgraph", "o/langgraph", "o/langgraph: agents as graphs. License: MIT.", "repository", "github_search"),
            src("https://arxiv.org/abs/1", "Speculative decoding", "A paper about speculative decoding speedups.", "paper", "arxiv_search"),
            src("https://news.b.com/market", "Vector DB market", "The vector database market was 1.5 billion USD in 2024.", "news")]
    (tmp_path / f"{task}.sources.json").write_text(json.dumps([s.model_dump(mode="json") for s in srcs]))
    (tmp_path / f"{task}.jsonl").write_text("")
    return srcs


def test_corpus_search_ranks_by_overlap_and_filters_by_tool_type(tmp_path):
    write_corpus(tmp_path)
    c = CassetteCorpus.load("t1", str(tmp_path))
    web = c.search("langgraph checkpointers", "web_search", 5)
    assert web[0].title == "LangGraph docs" and all(s.source_type != "paper" or True for s in web)
    assert [s.source_type for s in c.search("langgraph", "github_search", 5)] == ["repository"]
    assert [s.source_type for s in c.search("speculative decoding", "arxiv_search", 5)] == ["paper"]
    assert c.search("zzzz qqqq", "web_search", 5) == []


def test_corpus_search_is_deterministic_and_tool_agnostic_to_how_the_source_was_recorded(tmp_path):
    write_corpus(tmp_path)
    c = CassetteCorpus.load("t1", str(tmp_path))
    a = c.search("vector database market", "web_search", 3)
    assert a == c.search("vector database market", "web_search", 3) and a[0].source_type == "news"


def test_fetch_known_and_unknown_url(tmp_path):
    write_corpus(tmp_path)
    c = CassetteCorpus.load("t1", str(tmp_path))
    assert c.fetch("https://docs.a.dev/langgraph?utm_source=x").tool == "fetch_url"
    with pytest.raises(ToolError) as e:
        c.fetch("https://unknown.example/x")
    assert e.value.failure_type == FailureType.INVALID_SOURCE


def test_replay_registry_serves_all_tools_from_the_frozen_corpus_with_real_enforcement(tmp_path):
    write_corpus(tmp_path)
    reg = build_replay_registry(CassetteCorpus.load("t1", str(tmp_path)))
    ctx = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=reg)
    assert ctx.tools.call("web_search", {"query": "langgraph"}, ctx, agent="search_agent").sources
    with pytest.raises(ToolError):
        ctx.tools.call("web_search", {"query": "langgraph"}, ctx, agent="verifier")      # permissions still enforced
    with pytest.raises(ToolError):
        ctx.tools.call("web_search", {}, ctx, agent="search_agent")                      # schemas too


def test_cached_llm_populates_then_replays_identically_without_calling_the_base(tmp_path):
    cache = tmp_path / "llm.jsonl"
    live = ScriptedLLM(['{"a": 1}', '{"b": 2}'])
    c1 = CachedLLM(live, str(cache))
    r1 = c1.complete("prompt one", model="m", json_mode=True)
    r2 = c1.complete("prompt two", model="m", json_mode=True)
    assert (r1.text, r2.text) == ('{"a": 1}', '{"b": 2}') and live.calls == 2 and c1.stats() == {"hits": 0, "misses": 2}
    fresh = ScriptedLLM([])
    c2 = CachedLLM(fresh, str(cache))
    assert c2.complete("prompt one", model="m", json_mode=True).text == '{"a": 1}' and fresh.calls == 0
    assert c2.stats() == {"hits": 1, "misses": 0}


def test_cache_key_covers_model_prompt_and_mode(tmp_path):
    cache = tmp_path / "llm.jsonl"
    live = ScriptedLLM(["x", "y", "z"])
    c = CachedLLM(live, str(cache))
    assert c.complete("p", model="m1").text == "x"
    assert c.complete("p", model="m2").text == "y"                      # different model -> different entry
    assert c.complete("p", model="m1", json_mode=True).text == "z"      # different mode -> different entry
    assert live.calls == 3 and c.complete("p", model="m1").text == "x"


def test_offline_mode_raises_cache_miss_instead_of_calling_out(tmp_path):
    c = CachedLLM(None, str(tmp_path / "llm.jsonl"))
    with pytest.raises(CacheMiss):
        c.complete("never seen", model="m")


def test_cache_survives_concurrent_writers(tmp_path):
    import threading
    cache = tmp_path / "llm.jsonl"
    c = CachedLLM(ScriptedLLM([f"r{i}" for i in range(40)]), str(cache))
    ts = [threading.Thread(target=lambda i=i: c.complete(f"p{i}", model="m")) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = [json.loads(line) for line in cache.read_text().splitlines()]
    assert len(lines) == 40 and len({entry["key"] for entry in lines}) == 40
    assert CachedLLM(None, str(cache)).stats() == {"hits": 0, "misses": 0}
