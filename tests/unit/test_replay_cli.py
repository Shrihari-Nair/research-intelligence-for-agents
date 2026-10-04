import json
import time
from pathlib import Path

import pytest

from research_os.agents import build_agents
from research_os.agents.supervisor import Supervisor
from research_os.api.service import RunService
from research_os.contracts import RouterOutput, RoutingMode
from research_os.harness.replay import normalize_report, replay_main
from research_os.llm.demo_client import DemoLLM
from research_os.tools import build_registry


class Router:
    def route(self, text):
        return RouterOutput(intent="TECHNOLOGY_COMPARISON", confidence=0.95, routing_mode=RoutingMode.DIRECT, workflow="technology_comparison", latency_ms=1.0,
                            entities={"technologies": ["LangGraph", "CrewAI"], "companies": [], "topics": [], "time_scope": []})


def svc(tmp_path, llm=DemoLLM, tools=None):
    return RunService(router=Router(), supervisor=Supervisor(), agents=build_agents(), llm_factory=llm, tools_factory=tools or (lambda: build_registry(mode="fixture")),
                      runs_dir=str(tmp_path), llm_backend="demo", max_workers=2)


def finish(s, text="Compare LangGraph and CrewAI"):
    rec = s.submit(text)
    t0 = time.time()
    while not rec.done and time.time() - t0 < 30:
        time.sleep(0.05)
    assert rec.status.value == "SUCCESS", rec.status
    return rec


class Exploding:
    """An LLM that must never be called during replay."""

    def complete(self, *a, **k):
        raise AssertionError("the LLM must not be called during replay")


def exploding_tools():
    reg = build_registry(mode="fixture")
    for spec in reg._tools.values():
        spec.fn = lambda args, ctx: (_ for _ in ()).throw(AssertionError("a tool must not run during replay"))
    return reg


def test_every_run_is_recorded_with_its_report_and_config(tmp_path):
    rec = finish(svc(tmp_path))
    d = tmp_path / rec.run_id
    assert (d / "recording.jsonl").exists() and (d / "events.jsonl").exists()
    meta = json.loads((d / "run.json").read_text())
    assert meta["request"] == "Compare LangGraph and CrewAI" and meta["report"]["status"] == "SUCCESS" and meta["llm_backend"] == "demo"


def test_replay_reproduces_the_report_without_running_any_tool_or_llm(tmp_path):
    rec = finish(svc(tmp_path))
    result = svc(tmp_path, llm=Exploding, tools=exploding_tools).replay(rec.run_id)
    assert result["identical"] is True and result["differences"] == []
    assert result["n_calls_replayed"] > 5


def test_replay_detects_a_tampered_recording(tmp_path):
    rec = finish(svc(tmp_path))
    p = tmp_path / rec.run_id / "recording.jsonl"
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    for e in lines:
        if e["kind"] == "llm" and "synthesis" in json.dumps(e["args"])[:2000].lower() or (e["kind"] == "llm" and "findings section" in e["args"]["prompt"]):
            r = e["result"]
            r["text"] = r["text"].replace("verified statement", "TAMPERED statement")
    p.write_text("".join(json.dumps(e) + "\n" for e in lines))
    result = svc(tmp_path, llm=Exploding, tools=exploding_tools).replay(rec.run_id)
    assert result["identical"] is False and any("executive_summary" in d or "findings" in d for d in result["differences"])


def test_replay_miss_is_reported_not_swallowed(tmp_path):
    rec = finish(svc(tmp_path))
    p = tmp_path / rec.run_id / "recording.jsonl"
    p.write_text("")                                             # nothing recorded
    result = svc(tmp_path, llm=Exploding, tools=exploding_tools).replay(rec.run_id)
    assert result["identical"] is False and result["error"] and "no recorded" in result["error"].lower()


def test_unknown_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        svc(tmp_path).replay("nope")


def test_normalize_report_drops_only_timestamps_and_elapsed_time():
    rep = {"question": "q", "run_metadata": {"generated_at": "t", "budget": {"elapsed_seconds": 1.2, "steps": 5}, "run_id": "r"}, "findings": [1]}
    n = normalize_report(rep)
    assert n == {"question": "q", "run_metadata": {"budget": {"steps": 5}, "run_id": "r"}, "findings": [1]}
    assert rep["run_metadata"]["generated_at"] == "t"       # the input was not mutated


def test_replay_main_cli_exit_codes(tmp_path, monkeypatch, capsys):
    rec = finish(svc(tmp_path))
    monkeypatch.setattr("research_os.harness.replay.build_replay_service", lambda runs_dir: svc(Path(runs_dir), llm=Exploding, tools=exploding_tools))
    assert replay_main([rec.run_id, "--runs-dir", str(tmp_path)]) == 0
    assert "IDENTICAL" in capsys.readouterr().out
    (tmp_path / rec.run_id / "recording.jsonl").write_text("")
    assert replay_main([rec.run_id, "--runs-dir", str(tmp_path)]) == 1
    assert "DIFFERENT" in capsys.readouterr().out
