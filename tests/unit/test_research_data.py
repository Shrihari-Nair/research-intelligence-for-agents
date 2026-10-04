import json

import pandas as pd
import pytest

from research_os.common import load_intents
from research_os.router.data.build import build_dataset, clean_llm_rows, dedupe_near
from research_os.router.data.load import load_split
from research_os.router.data.research_data import generate_templates
from research_os.router.data.validate import normalize

INTENTS = load_intents()


def test_templates_cover_all_intents_unique_and_deterministic():
    a, b = generate_templates(7, 12), generate_templates(7, 12)
    assert a.equals(b)
    assert set(a["intent"]) == set(INTENTS) and (a["intent"].value_counts() == 12).all()
    assert not a["text"].map(normalize).duplicated().any()
    assert a["text"].map(lambda t: 3 <= len(t.split()) <= 35).all()


def test_dedupe_near_removes_close_variants_keeps_first():
    df = pd.DataFrame({"text": ["compare langgraph and crewai for agents", "compare langgraph and crewai for agents please",
                                "who created fastapi"], "intent": ["A", "A", "B"]})
    out = dedupe_near(df, 0.8)
    assert out["text"].tolist() == ["compare langgraph and crewai for agents", "who created fastapi"]


def test_clean_drops_literature_without_research_cue_and_monitoring_without_ongoing_cue():
    df = pd.DataFrame({"text": ["what's new in computer vision", "find recent papers on vision", "what are cohere's rivals",
                                "monitor new vision releases", "tell me about vision releases"],
                       "intent": ["LITERATURE_REVIEW", "LITERATURE_REVIEW", "COMPETITIVE_INTELLIGENCE",
                                  "RESEARCH_MONITORING", "RESEARCH_MONITORING"]})
    out, dropped = clean_llm_rows(df)
    assert out["text"].tolist() == ["find recent papers on vision", "what are cohere's rivals", "monitor new vision releases"]
    assert dropped == {"LITERATURE_REVIEW": 1, "RESEARCH_MONITORING": 1}


def test_build_dataset_is_balanced_leak_free_and_excludes_hard_test(tmp_path):
    llm = pd.DataFrame({"text": [f"find papers on topic number {i} please" for i in range(20)], "intent": "LITERATURE_REVIEW"})
    llm_dir = tmp_path / "ex"
    llm_dir.mkdir()
    llm.to_json(llm_dir / "llm_generated.jsonl", orient="records", lines=True)
    hard = tmp_path / "hard.jsonl"
    hard.write_text("\n".join(json.dumps({"text": t, "intent": "FACT_LOOKUP"}) for t in ["who created fastapi"]) + "\n")
    cfg = {"seed": 1, "data": {"template_per_intent": 30, "max_per_intent": 30, "llm_examples_dir": str(llm_dir),
                               "hard_test": str(hard), "near_duplicate_jaccard": 0.9},
           "split": {"train": 0.7, "val": 0.15, "test": 0.15}, "paths": {"intents": "configs/intents.yaml"}}
    res = build_dataset(cfg, INTENTS)
    parts = {k: res[k] for k in ("train", "val", "test")}
    assert (res["all"]["intent"].value_counts() == res["all"]["intent"].value_counts().iloc[0]).all()
    sets = {k: set(map(normalize, v["text"])) for k, v in parts.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert "who created fastapi" not in set(map(normalize, res["all"]["text"]))
    assert res["stats"]["per_intent"]["LITERATURE_REVIEW"]["llm"] > 0


# ---- the committed dataset ----

def _split(name):
    try:
        return load_split(name, "data")
    except FileNotFoundError:
        pytest.skip("run `make data` first")


def test_committed_splits_are_leak_free_incl_hard_test():
    parts = {n: set(map(normalize, _split(n)["text"])) for n in ("train", "validation", "test", "hard_test")}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (parts[a] & parts[b]), f"leak between {a} and {b}"


def test_committed_hard_test_is_valid_and_covers_every_intent():
    h = _split("hard_test")
    assert set(h["intent"]) <= set(INTENTS) and 55 <= len(h) <= 70
    assert (h["intent"].value_counts() >= 5).all() and len(set(h["intent"])) == 8


def test_committed_training_data_is_balanced_and_sized():
    tr = _split("train")
    per = tr["intent"].value_counts()
    assert per.min() >= 60 and (per.max() - per.min()) <= 2
