from research_os.legacy.agents.tools import ToolBox
from research_os.legacy.graph.workflow import build_graph
from research_os.voice.legacy_adapters import TextSTT, TextTTS, voice_turn


class Stub:
    def route(self, text):
        return {"intent": "GENERAL_QUERY", "confidence": 0.9, "source": "minilm", "llm_called": False,
                "llm_failed": False, "latency_ms": 1.0, "input_tokens": 0, "output_tokens": 0}


def test_text_adapters_round_trip_through_graph():
    graph = build_graph(Stub(), ToolBox(), None)
    tts = TextTTS()
    out = voice_turn(TextSTT(), graph, tts, "what are your opening hours")
    assert "open" in out["response"].lower()
    assert tts.spoken == [out["response"]]


def test_empty_transcript_is_not_routed():
    graph = build_graph(Stub(), ToolBox(), None)
    tts = TextTTS()
    out = voice_turn(TextSTT(), graph, tts, "   ")
    assert out is None and tts.spoken == []
