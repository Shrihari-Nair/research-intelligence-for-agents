import pytest
from fastapi.testclient import TestClient

from research_os.common import load_intents
from research_os.legacy.agents.tools import ToolBox
from research_os.legacy.api.app import create_app
from research_os.legacy.graph.workflow import build_graph
from research_os.llm.mock_client import MockLLMClient
from research_os.router.systems.hybrid import HybridSystem

INTENTS = load_intents("configs/healthcare/intents.yaml")


class KeywordRouter:
    """Cheap stand-in for the SetFit model so API tests need no training."""

    def predict(self, text):
        low = text.lower()
        intent, conf = "GENERAL_QUERY", 0.9
        if "move" in low or "reschedule" in low:
            intent = "RESCHEDULE_APPOINTMENT"
        elif "chest" in low:
            intent = "EMERGENCY"
        elif "weird" in low:
            conf = 0.3
        probs = {i: 0.01 for i in INTENTS}
        probs[intent] = conf
        return {"intent": intent, "confidence": conf, "probs": probs}


@pytest.fixture()
def client():
    llm = MockLLMClient()
    system = HybridSystem(KeywordRouter(), llm, {"threshold": 0.8, "emergency_floor": 0.3}, INTENTS)
    graph = build_graph(system, ToolBox(), llm)
    with TestClient(create_app(graph=graph, llm_backend="mock", max_chars=100)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_chat_happy_path(client):
    r = client.post("/chat", json={"text": "Can I move my appointment to Friday?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"intent", "confidence", "source", "agent", "response", "latency_ms"}
    assert body["intent"] == "RESCHEDULE_APPOINTMENT" and body["agent"] == "booking_agent"
    assert body["source"] == "minilm" and 0 <= body["confidence"] <= 1


def test_chat_low_confidence_uses_llm_source(client):
    body = client.post("/chat", json={"text": "something weird please cancel my visit"}).json()
    assert body["source"] == "llm" and body["intent"] == "CANCEL_APPOINTMENT"


def test_chat_emergency(client):
    body = client.post("/chat", json={"text": "my chest hurts"}).json()
    assert body["agent"] == "emergency_agent" and body["source"] == "emergency_rule"


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "    "}, {}, {"text": "x" * 101}, {"text": 5}])
def test_chat_validation_errors(client, payload):
    assert client.post("/chat", json=payload).status_code == 422
