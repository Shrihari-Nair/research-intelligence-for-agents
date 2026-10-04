import time

import pytest

from research_os.agents.supervisor import Supervisor
from research_os.api.service import RunService
from research_os.contracts import RouterOutput, RoutingMode, Source
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.voice.adapters import TextSTT, TextTTS, spoken_summary, voice_turn


class Router:
    def route(self, text):
        return RouterOutput(intent="GENERAL_RESEARCH", confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow="general_research", latency_ms=1,
                            entities={"technologies": [], "companies": [], "topics": [], "time_scope": []})


class Agent:
    SUMMARY = "Raft elects a leader using randomized timeouts. A candidate that wins a majority becomes leader. Followers then accept heartbeats. Extra sentence."

    def run(self, step, inputs, ctx, request):
        if step.agent == "search_agent":
            from datetime import datetime, timezone
            ctx.sources.add(Source(source_id="s1", title="Raft", url="https://a.dev/raft", source_type="documentation",
                                   retrieved_at=datetime.now(timezone.utc), content="q", tool="web_search"))
        if step.agent == "verifier":
            c = {"claim_id": "c1", "claim": "A claim.", "source_id": "s1", "evidence_quote": "q", "extractor_confidence": 0.9, "verdict": "SUPPORTED",
                 "verdict_rationale": "ok"}
            return {"claims": [c], "conflicts": [], "gate": {"result": "PASS", "reason": None, "failure_type": None}}
        if step.agent == "synthesis_agent":
            return {"summary": self.SUMMARY, "findings": [{"statement": "A claim.", "claim_ids": ["c1"]}], "claim_ids": ["c1"]}
        return {"ok": step.id}


def service(tmp_path, router=None):
    ag = Agent()
    return RunService(router=router or Router(), supervisor=Supervisor(), agents={n: ag for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")},
                      llm_factory=lambda: ScriptedLLM([]), tools_factory=lambda: build_registry(mode="fixture"), runs_dir=str(tmp_path), llm_backend="mock")


def test_spoken_summary_is_short_plain_and_sentence_bounded():
    s = spoken_summary(Agent.SUMMARY, max_sentences=2, max_words=40)
    assert s == "Raft elects a leader using randomized timeouts. A candidate that wins a majority becomes leader."
    assert spoken_summary("One long sentence " + "word " * 80, max_sentences=3, max_words=20).endswith("…")
    assert "[" not in spoken_summary("See [1] and [2] for details. Done.", 3, 40) and "**" not in spoken_summary("**Bold** claim here.", 3, 40)
    assert spoken_summary("", 2, 40) == ""


def test_voice_turn_runs_the_pipeline_and_speaks_a_summary_with_a_report_link(tmp_path):
    svc, tts = service(tmp_path), TextTTS()
    out = voice_turn(TextSTT(), svc, tts, "tell me about raft consensus", base_url="http://localhost:8000", timeout_s=10)
    assert out["status"] == "SUCCESS" and out["report_url"].startswith("http://localhost:8000/runs/") and out["report_url"].endswith("/report?format=pdf")
    assert tts.spoken == [out["spoken"]] and "Raft elects a leader" in out["spoken"] and "report" in out["spoken"].lower()
    assert len(out["spoken"].split()) < 90


def test_empty_transcripts_are_ignored_and_bytes_are_decoded(tmp_path):
    svc, tts = service(tmp_path), TextTTS()
    assert voice_turn(TextSTT(), svc, tts, "   ", base_url="x") is None and tts.spoken == []
    assert voice_turn(TextSTT(), svc, tts, b"tell me about raft", base_url="x", timeout_s=10)["status"] == "SUCCESS"


def test_a_failed_run_is_spoken_honestly_not_as_a_success(tmp_path):
    class Boom:
        def route(self, text):
            raise RuntimeError("router exploded")
    svc, tts = service(tmp_path, router=Boom()), TextTTS()
    out = voice_turn(TextSTT(), svc, tts, "tell me about raft", base_url="x", timeout_s=10)
    assert out["status"] == "FAILED" and "couldn't" in out["spoken"].lower() and "router exploded" not in out["spoken"]


def test_voice_never_changes_the_core_pipeline_inputs(tmp_path):
    svc = service(tmp_path)
    rec_text = []
    orig = svc.submit
    svc.submit = lambda text, profile=None: rec_text.append(text) or orig(text, profile)
    voice_turn(TextSTT(), svc, TextTTS(), "  Tell me about raft  ", base_url="x", timeout_s=10)
    assert rec_text == ["Tell me about raft"]                      # the transcript goes in as a normal request, trimmed


def test_a_timeout_is_reported_as_still_running(tmp_path):
    class Slow(Agent):
        def run(self, step, inputs, ctx, request):
            if step.agent == "search_agent":
                time.sleep(2)
            return super().run(step, inputs, ctx, request)
    svc = service(tmp_path)
    svc.agents = {n: Slow() for n in svc.agents}
    out = voice_turn(TextSTT(), svc, TextTTS(), "tell me about raft", base_url="x", timeout_s=0.3)
    assert out["status"] in ("RUNNING", "PENDING") and "still working" in out["spoken"].lower()


def test_livekit_adapter_fails_clearly_when_the_optional_extra_is_missing():
    from research_os.voice.livekit_agent import LiveKitUnavailable, require_livekit
    try:
        import livekit  # noqa: F401
        pytest.skip("livekit is installed")
    except ImportError:
        with pytest.raises(LiveKitUnavailable, match="requirements-voice.txt"):
            require_livekit()
