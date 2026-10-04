import pandas as pd
import pytest

from research_os.common import load_intents
from research_os.llm.mock_client import MockLLMClient
from research_os.router.evaluation.compare_systems import cost_per_request, run_e2e, run_routing_eval
from research_os.router.evaluation.e2e_tasks import E2E_TASKS, task_success
from research_os.router.systems.minilm_only import MiniLMOnlySystem

PRICING = {"input": 1.0, "output": 2.0}


class FixedRouter:
    def __init__(self, intent):
        self.intent = intent

    def predict(self, text):
        probs = {i: 0.0 for i in load_intents("configs/healthcare/intents.yaml")}
        probs[self.intent] = 1.0
        return {"intent": self.intent, "confidence": 1.0, "probs": probs}


def test_e2e_tasks_are_well_formed():
    assert 36 <= len(E2E_TASKS) <= 48
    intents = set(load_intents("configs/healthcare/intents.yaml"))
    hard = set(pd.read_csv("data/healthcare/hard_test.csv")["text"])
    assert {t["expected_intent"] for t in E2E_TASKS} == intents
    assert sum(t["text"] in hard for t in E2E_TASKS) >= 8
    for t in E2E_TASKS:
        assert set(t) == {"text", "expected_intent", "expected_agent", "expected_tool"}


def test_task_success_requires_agent_and_tool():
    task = {"text": "x", "expected_intent": "CANCEL_APPOINTMENT", "expected_agent": "booking_agent",
            "expected_tool": "cancel_appointment"}
    ok = {"agent_name": "booking_agent", "tool_calls": [{"name": "cancel_appointment", "args": {}}]}
    assert task_success(ok, task)
    assert not task_success({**ok, "agent_name": "faq_agent"}, task)
    assert not task_success({**ok, "tool_calls": []}, task)


def test_task_success_no_tool_expected_means_no_tools_called():
    task = {"text": "x", "expected_intent": "EMERGENCY", "expected_agent": "emergency_agent",
            "expected_tool": None}
    assert task_success({"agent_name": "emergency_agent", "tool_calls": []}, task)
    assert not task_success({"agent_name": "emergency_agent", "tool_calls": [{"name": "t"}]}, task)


def test_cost_per_request_arithmetic():
    recs = [{"input_tokens": 1_000_000, "output_tokens": 0}, {"input_tokens": 0, "output_tokens": 500_000}]
    assert cost_per_request(recs, PRICING) == pytest.approx((1.0 + 1.0) / 2)
    assert cost_per_request([{"input_tokens": 0, "output_tokens": 0}], PRICING) == 0.0


def test_run_routing_eval_reports_all_fields():
    df = pd.DataFrame({"text": ["a b c", "d e f"], "intent": ["CANCEL_APPOINTMENT", "GENERAL_QUERY"]})
    res = run_routing_eval(MiniLMOnlySystem(FixedRouter("CANCEL_APPOINTMENT"),
                                            {"emergency_floor": 0.3}), df, load_intents("configs/healthcare/intents.yaml"), PRICING)
    assert res["accuracy"] == pytest.approx(0.5)
    for k in ("macro_f1", "p50_ms", "p95_ms", "cpu_ms_per_request", "cost_per_request", "llm_calls",
              "emergency_recall", "n"):
        assert k in res
    assert res["llm_calls"] == 0 and res["cost_per_request"] == 0.0


def test_run_e2e_counts_successes():
    from research_os.legacy.agents.tools import ToolBox
    tasks = [t for t in E2E_TASKS if t["expected_intent"] == "GENERAL_QUERY"][:2]
    res = run_e2e(MiniLMOnlySystem(FixedRouter("GENERAL_QUERY"), {"emergency_floor": 0.3}), tasks,
                  MockLLMClient(), ToolBox)
    assert res["n"] == 2 and res["task_success"] == 1.0
