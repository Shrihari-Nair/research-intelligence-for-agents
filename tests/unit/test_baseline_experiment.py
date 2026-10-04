import json

import pandas as pd

from research_os.common import load_intents
from research_os.router.evaluation.baseline_experiment import run_experiment
from research_os.router.systems.minilm_only import MiniLMOnlySystem

INTENTS = load_intents("configs/healthcare/intents.yaml")


class FixedRouter:
    def predict(self, text):
        probs = {i: 0.0 for i in INTENTS}
        probs["CANCEL_APPOINTMENT"] = 1.0
        return {"intent": "CANCEL_APPOINTMENT", "confidence": 1.0, "probs": probs}


def test_experiment_writes_all_artifacts_and_marks_missing_llm(tmp_path):
    df = pd.DataFrame({"text": ["cancel it", "what time do you open"],
                       "intent": ["CANCEL_APPOINTMENT", "GENERAL_QUERY"]})
    systems = {"minilm_only": MiniLMOnlySystem(FixedRouter(), {"emergency_floor": 0.3}), "llm_only": None}
    res = run_experiment(systems, df, INTENTS, {"input": 1.0, "output": 2.0}, str(tmp_path), seed=42)
    assert res["systems"]["llm_only"] == "not_run"
    m = res["systems"]["minilm_only"]
    assert m["accuracy"] == 0.5 and m["per_intent_f1"]["CANCEL_APPOINTMENT"] > 0
    assert m["input_tokens"] == 0 and m["llm_calls"] == 0
    assert res["seed"] == 42 and "hardware" in res
    assert (tmp_path / "baseline_results.json").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    rows = pd.read_csv(tmp_path / "routing_benchmark.csv")
    assert set(rows["system"]) == {"minilm_only", "llm_only"}
    assert json.loads((tmp_path / "baseline_results.json").read_text())["split"] == "test"
