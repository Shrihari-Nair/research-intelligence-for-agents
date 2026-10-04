import pytest

from research_os.common import load_intents
from research_os.legacy.agents.tools import ToolBox
from research_os.legacy.graph.workflow import INTENT_TO_AGENT, build_graph, run_text
from research_os.llm.mock_client import MockLLMClient

EXPECTED = {
    "BOOK_APPOINTMENT": ("booking_agent", "book_appointment", "I want to see Dr Sharma on friday"),
    "RESCHEDULE_APPOINTMENT": ("booking_agent", "reschedule_appointment", "Can I move my appointment to Friday?"),
    "CANCEL_APPOINTMENT": ("booking_agent", "cancel_appointment", "cancel my appointment"),
    "PRESCRIPTION_QUERY": ("prescription_agent", "get_prescription", "what medicines did the doctor give me"),
    "FOLLOWUP_QUERY": ("prescription_agent", "get_followup", "when should I follow up"),
    "MEDICAL_RESEARCH": ("research_agent", "search_pubmed", "latest treatments for migraine"),
    "GENERAL_QUERY": ("faq_agent", None, "what are your opening hours"),
    "EMERGENCY": ("emergency_agent", None, "I'm having severe chest pain"),
}


class StubSystem:
    def __init__(self, intent):
        self.intent = intent

    def route(self, text):
        return {"intent": self.intent, "confidence": 0.99, "source": "minilm", "llm_called": False,
                "llm_failed": False, "latency_ms": 1.0, "input_tokens": 0, "output_tokens": 0}


def test_every_intent_has_an_agent():
    assert set(INTENT_TO_AGENT) == set(load_intents("configs/healthcare/intents.yaml"))


@pytest.mark.parametrize("intent", list(EXPECTED))
def test_scenario_per_intent(intent):
    agent, tool, text = EXPECTED[intent]
    g = build_graph(StubSystem(intent), ToolBox(), MockLLMClient())
    state = run_text(g, text)
    assert state["agent_name"] == agent
    assert state["routing"]["intent"] == intent
    names = [c["name"] for c in state["tool_calls"]]
    if tool is None:
        assert names == []
    else:
        assert tool in names
    assert state["response"]


def test_router_node_is_pluggable():
    a = run_text(build_graph(StubSystem("CANCEL_APPOINTMENT"), ToolBox(), None), "x")
    b = run_text(build_graph(StubSystem("GENERAL_QUERY"), ToolBox(), None), "x")
    assert a["agent_name"] != b["agent_name"]
