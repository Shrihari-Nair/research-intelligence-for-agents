import json
import threading

import pytest

from research_os.contracts import FailureType, Source
from research_os.harness.context import RunContext, SourceStore
from research_os.harness.events import EventEmitter
from research_os.harness.replay import Recorder, ReplayMiss
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools.registry import ToolError, ToolRegistry, ToolResult


def make_source(url="https://a.com/x", content="hello world"):
    from datetime import datetime, timezone

    from research_os.contracts import source_id_for_url
    return Source(source_id=source_id_for_url(url), title="t", url=url, source_type="documentation",
                  retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc), content=content, tool="web_search")


def test_events_have_monotonic_seq_and_jsonl_sink(tmp_path):
    em = EventEmitter("r1", sink_dir=str(tmp_path))
    em.emit("router", "started", {"a": 1})
    em.emit("router", "finished")
    assert [e.seq for e in em.events] == [0, 1]
    lines = (tmp_path / "r1" / "events.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [0, 1]


def test_events_since_returns_tail_for_sse_resume():
    em = EventEmitter("r1")
    for i in range(5):
        em.emit("n", "info", {"i": i})
    assert [e.seq for e in em.since(3)] == [3, 4]


def test_event_seq_is_unique_under_threads():
    em = EventEmitter("r1")
    ts = [threading.Thread(target=lambda: [em.emit("n", "info") for _ in range(50)]) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(e.seq for e in em.events) == list(range(200))


def test_recorder_records_then_replays_identically(tmp_path):
    rec = Recorder("record", str(tmp_path / "rec.jsonl"))
    out = rec.call("tool", "web_search", {"q": "x"}, lambda: {"results": [1, 2]})
    rec.save()
    rep = Recorder("replay", str(tmp_path / "rec.jsonl"))
    assert rep.call("tool", "web_search", {"q": "x"}, lambda: pytest.fail("must not execute")) == out


def test_replay_miss_is_loud_and_repeated_calls_replay_in_order(tmp_path):
    rec = Recorder("record", str(tmp_path / "r.jsonl"))
    seq = iter([{"n": 1}, {"n": 2}])
    rec.call("tool", "t", {"a": 1}, lambda: next(seq))
    rec.call("tool", "t", {"a": 1}, lambda: next(seq))
    rec.save()
    rep = Recorder("replay", str(tmp_path / "r.jsonl"))
    assert rep.call("tool", "t", {"a": 1}, lambda: None) == {"n": 1}
    assert rep.call("tool", "t", {"a": 1}, lambda: None) == {"n": 2}
    with pytest.raises(ReplayMiss):
        rep.call("tool", "t", {"a": 1}, lambda: None)
    with pytest.raises(ReplayMiss):
        rep.call("tool", "other", {}, lambda: None)


def test_recorder_records_and_replays_errors(tmp_path):
    rec = Recorder("record", str(tmp_path / "e.jsonl"))

    def boom():
        raise ToolError(FailureType.SEARCH_TIMEOUT, "slow")
    with pytest.raises(ToolError):
        rec.call("tool", "web_search", {"q": "x"}, boom)
    rec.save()
    rep = Recorder("replay", str(tmp_path / "e.jsonl"))
    with pytest.raises(ToolError) as e:
        rep.call("tool", "web_search", {"q": "x"}, lambda: None)
    assert e.value.failure_type == FailureType.SEARCH_TIMEOUT


def test_registry_dispatches_records_counts_and_emits(tmp_path):
    reg = ToolRegistry()
    reg.register("web_search", lambda args, ctx: ToolResult(sources=[make_source()]), cost_class="search")
    ctx = RunContext.create("r1", {}, llm=ScriptedLLM([]), tools=reg, runs_dir=str(tmp_path))
    res = ctx.tools.call("web_search", {"q": "x"}, ctx)
    assert res.sources[0].url == "https://a.com/x"
    assert ctx.budget.usage()["tool_calls"] == 1 and ctx.budget.usage()["searches"] == 1
    assert [e.type for e in ctx.events.events if e.node == "tool:web_search"] == ["started", "finished"]
    assert ctx.sources.get(res.sources[0].source_id) is not None


def test_registry_unknown_tool_is_a_tool_error():
    ctx = RunContext.create("r1", {}, llm=ScriptedLLM([]), tools=ToolRegistry())
    with pytest.raises(ToolError) as e:
        ctx.tools.call("nope", {}, ctx)
    assert e.value.failure_type == FailureType.TOOL_FAILURE


def test_registry_wraps_unexpected_exceptions_as_tool_failure():
    reg = ToolRegistry()
    reg.register("bad", lambda args, ctx: 1 / 0)
    ctx = RunContext.create("r1", {}, llm=ScriptedLLM([]), tools=reg)
    with pytest.raises(ToolError) as e:
        ctx.tools.call("bad", {}, ctx)
    assert e.value.failure_type == FailureType.TOOL_FAILURE
    assert [ev.type for ev in ctx.events.events if ev.node == "tool:bad"] == ["started", "failed"]


def test_source_store_dedupes_by_id():
    st = SourceStore()
    assert st.add(make_source()) is True and st.add(make_source()) is False
    assert len(st.all()) == 1


def test_context_llm_records_counts_and_replays(tmp_path):
    base = ScriptedLLM(['{"a": 1}'])
    ctx = RunContext.create("r1", {}, llm=base, tools=ToolRegistry(), runs_dir=str(tmp_path), mode="record")
    r = ctx.llm.complete("p", json_mode=True)
    assert r.text == '{"a": 1}' and ctx.budget.usage()["llm_calls"] == 1
    ctx.recorder.save()
    rep = RunContext.create("r2", {}, llm=ScriptedLLM([]), tools=ToolRegistry(), runs_dir=str(tmp_path), mode="replay",
                            recording_path=ctx.recorder.path)
    assert rep.llm.complete("p", json_mode=True).text == '{"a": 1}'
