import pytest

from research_os.common import load_intents
from research_os.llm.client import FALLBACK_INTENT, parse_llm_intent
from research_os.llm.gemini_client import GeminiLLMClient
from research_os.llm.mock_client import MockLLMClient

INTENTS = load_intents("configs/healthcare/intents.yaml")


def test_parse_valid_json():
    r = parse_llm_intent('{"intent": "CANCEL_APPOINTMENT", "reason": "wants to cancel"}', INTENTS)
    assert r["intent"] == "CANCEL_APPOINTMENT" and r["failed"] is False


def test_parse_fenced_json():
    raw = '```json\n{"intent": "EMERGENCY", "reason": "chest pain"}\n```'
    assert parse_llm_intent(raw, INTENTS)["intent"] == "EMERGENCY"


@pytest.mark.parametrize("raw", ["I think it is a booking", "", '{"intent": "DANCE", "reason": "x"}',
                                 "{not json", '["a"]', '{"reason": "no intent"}'])
def test_parse_malformed_falls_back(raw):
    r = parse_llm_intent(raw, INTENTS)
    assert r["intent"] == FALLBACK_INTENT and r["failed"] is True


def test_mock_is_deterministic_and_never_raises():
    m = MockLLMClient()
    a = m.classify_intent("please cancel my appointment", INTENTS)
    b = m.classify_intent("please cancel my appointment", INTENTS)
    assert a == b and a["intent"] == "CANCEL_APPOINTMENT" and a["failed"] is False
    assert m.classify_intent("zzz", INTENTS)["intent"] in INTENTS
    assert m.calls == 3


def test_mock_marks_tokens_as_mock():
    assert MockLLMClient().classify_intent("hi", INTENTS)["mock"] is True


class _Boom:
    def generate(self, **_):
        raise RuntimeError("network down")


class _Resp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = type("U", (), {"prompt_token_count": 11, "candidates_token_count": 5})()


class _Ok:
    def generate(self, **_):
        return _Resp('{"intent": "FOLLOWUP_QUERY", "reason": "asks next visit"}')


def test_gemini_failure_falls_back():
    c = GeminiLLMClient(model="m", backend=_Boom())
    r = c.classify_intent("hello", INTENTS)
    assert r["intent"] == FALLBACK_INTENT and r["failed"] is True


def test_gemini_success_reports_tokens():
    r = GeminiLLMClient(model="m", backend=_Ok()).classify_intent("when do I return", INTENTS)
    assert r["intent"] == "FOLLOWUP_QUERY" and r["failed"] is False
    assert r["input_tokens"] == 11 and r["output_tokens"] == 5


def test_gemini_without_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GeminiLLMClient(model="m")


def test_gemini_default_key_env_matches_routing_config():
    import inspect

    from research_os.common import load_config
    default = inspect.signature(GeminiLLMClient.__init__).parameters["api_key_env"].default
    assert default == load_config("configs/models.yaml")["api_key_env"]
