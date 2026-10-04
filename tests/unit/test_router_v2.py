import pytest
from pydantic import ValidationError

from research_os.contracts.router import RouterOutput, RoutingMode
from research_os.router.entities import EntityExtractor
from research_os.router.inference import ResearchRouter
from research_os.router.workflows import workflow_for_intent, workflow_map

INTENTS = ["FACT_LOOKUP", "TECHNOLOGY_COMPARISON", "LITERATURE_REVIEW", "TECHNICAL_DEEP_DIVE",
           "COMPETITIVE_INTELLIGENCE", "MARKET_RESEARCH", "RESEARCH_MONITORING", "GENERAL_RESEARCH"]


class FakeIntentRouter:
    def __init__(self, intent, conf):
        self.intent, self.conf = intent, conf

    def predict(self, text):
        if not text.strip():
            raise ValueError("empty")
        probs = {i: (1 - self.conf) / 7 for i in INTENTS}
        probs[self.intent] = self.conf
        return {"intent": self.intent, "confidence": self.conf, "probs": probs}


def make(intent="TECHNOLOGY_COMPARISON", conf=0.9, threshold=0.5):
    return ResearchRouter(FakeIntentRouter(intent, conf), EntityExtractor.from_config(), threshold, workflow_map())


def test_every_intent_has_exactly_one_workflow():
    m = workflow_map()
    assert set(m) == set(INTENTS) and len(set(m.values())) == 8
    assert workflow_for_intent("FACT_LOOKUP") == "fact_lookup"


def test_unknown_intent_workflow_raises():
    with pytest.raises(KeyError):
        workflow_for_intent("NOPE")


def test_high_confidence_is_direct_with_entities_and_workflow():
    out = make(conf=0.9).route("Compare LangGraph and CrewAI")
    assert isinstance(out, RouterOutput) and out.routing_mode == RoutingMode.DIRECT
    assert out.intent == "TECHNOLOGY_COMPARISON" and out.workflow == "technology_comparison"
    assert out.entities["technologies"] == ["LangGraph", "CrewAI"]
    assert out.latency_ms > 0 and out.confidence == pytest.approx(0.9)


def test_low_confidence_falls_back_but_keeps_provisional_workflow():
    out = make(conf=0.3).route("hmm")
    assert out.routing_mode == RoutingMode.LLM_FALLBACK and out.workflow == "technology_comparison"


def test_threshold_boundary_is_inclusive():
    assert make(conf=0.5, threshold=0.5).route("x y z").routing_mode == RoutingMode.DIRECT


def test_missing_threshold_is_a_loud_error():
    with pytest.raises(ValueError, match="threshold"):
        make(threshold=None)


def test_empty_text_raises_value_error():
    with pytest.raises(ValueError):
        make().route("   ")


def test_contract_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        RouterOutput(intent="X", confidence=1.5, routing_mode="DIRECT", workflow="w", entities={}, latency_ms=1)
