import pytest

from research_os.common import load_intents
from research_os.llm.mock_client import MockLLMClient
from research_os.router.systems.hybrid import HybridSystem
from research_os.router.systems.minilm_only import MiniLMOnlySystem

INTENTS = load_intents("configs/healthcare/intents.yaml")
CFG = {"threshold": 0.80, "emergency_floor": 0.30}


class FakeRouter:
    """Returns scripted probabilities: {intent: prob}; remainder spread evenly."""

    def __init__(self, probs):
        rest = (1 - sum(probs.values())) / (len(INTENTS) - len(probs))
        self.p = {i: probs.get(i, rest) for i in INTENTS}

    def predict(self, text):
        top = max(self.p, key=self.p.get)
        return {"intent": top, "confidence": self.p[top], "probs": dict(self.p)}


class BoomLLM:
    def classify_intent(self, text, intents):
        raise RuntimeError("down")


class GarbageLLM:
    def classify_intent(self, text, intents):
        from research_os.llm.client import parse_llm_intent
        return {**parse_llm_intent("lol no json", intents), "input_tokens": 3, "output_tokens": 1}


def test_high_confidence_skips_llm():
    llm = MockLLMClient()
    r = HybridSystem(FakeRouter({"CANCEL_APPOINTMENT": 0.95}), llm, CFG).route("cancel it")
    assert r["intent"] == "CANCEL_APPOINTMENT" and r["source"] == "minilm"
    assert r["llm_called"] is False and llm.calls == 0
    assert r["latency_ms"] >= 0


def test_low_confidence_calls_llm():
    llm = MockLLMClient()
    r = HybridSystem(FakeRouter({"BOOK_APPOINTMENT": 0.45}), llm, CFG).route("please cancel my visit")
    assert r["source"] == "llm" and r["llm_called"] is True and llm.calls == 1
    assert r["intent"] == "CANCEL_APPOINTMENT"
    assert r["confidence"] == pytest.approx(0.45)  # router confidence retained for analysis


def test_emergency_top_intent_overrides_without_llm():
    llm = MockLLMClient()
    r = HybridSystem(FakeRouter({"EMERGENCY": 0.5}), llm, CFG).route("help")
    assert r["intent"] == "EMERGENCY" and r["source"] == "emergency_rule" and llm.calls == 0


def test_emergency_floor_overrides_even_when_other_intent_is_confident():
    llm = MockLLMClient()
    r = HybridSystem(FakeRouter({"GENERAL_QUERY": 0.65, "EMERGENCY": 0.32}), llm, CFG).route("x")
    assert r["intent"] == "EMERGENCY" and llm.calls == 0


def test_just_below_floor_does_not_trigger():
    r = HybridSystem(FakeRouter({"GENERAL_QUERY": 0.85, "EMERGENCY": 0.29}), MockLLMClient(), CFG).route("x")
    assert r["intent"] == "GENERAL_QUERY"


def test_malformed_llm_output_falls_back_to_general():
    r = HybridSystem(FakeRouter({"BOOK_APPOINTMENT": 0.4}), GarbageLLM(), CFG).route("x")
    assert r["intent"] == "GENERAL_QUERY" and r["llm_failed"] is True and r["llm_called"] is True


def test_llm_exception_falls_back_to_general():
    r = HybridSystem(FakeRouter({"BOOK_APPOINTMENT": 0.4}), BoomLLM(), CFG).route("x")
    assert r["intent"] == "GENERAL_QUERY" and r["llm_failed"] is True


def test_threshold_comes_from_config():
    llm = MockLLMClient()
    sys_ = HybridSystem(FakeRouter({"BOOK_APPOINTMENT": 0.6}), llm, {**CFG, "threshold": 0.5})
    assert sys_.route("x")["llm_called"] is False
    sys2 = HybridSystem(FakeRouter({"BOOK_APPOINTMENT": 0.6}), llm, {**CFG, "threshold": 0.7})
    assert sys2.route("x")["llm_called"] is True


def test_missing_threshold_raises():
    with pytest.raises(ValueError, match="threshold"):
        HybridSystem(FakeRouter({}), MockLLMClient(), {"emergency_floor": 0.3, "threshold": None})


def test_minilm_only_never_calls_llm_but_keeps_emergency_rule():
    s = MiniLMOnlySystem(FakeRouter({"GENERAL_QUERY": 0.6, "EMERGENCY": 0.35}), CFG)
    r = s.route("x")
    assert r["intent"] == "EMERGENCY" and r["llm_called"] is False
    r2 = MiniLMOnlySystem(FakeRouter({"BOOK_APPOINTMENT": 0.3}), CFG).route("x")
    assert r2["intent"] == "BOOK_APPOINTMENT" and r2["source"] == "minilm"
