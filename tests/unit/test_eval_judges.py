import json

import pytest

from evaluation.judges import Judge, agreement
from research_os.llm.mock_client import ScriptedLLM
from research_os.llm.structured import MalformedLLMOutput


def labels(*pairs):
    return json.dumps({"labels": [{"id": i, "label": v} for i, v in pairs]})


def ratings(*pairs):
    return json.dumps({"ratings": [{"id": i, "score": s} for i, s in pairs]})


ITEMS = [{"id": "a-long-hash-1", "claim": "X is fast", "quote": "X is fast", "source": "S1"},
         {"id": "b-long-hash-2", "claim": "Y is best", "quote": "Y exists", "source": "S2"}]


def test_claim_support_labels_by_id_and_uses_the_eval_judge_model():
    llm = ScriptedLLM([labels(("k1", "SUPPORTED"), ("k2", "UNSUPPORTED"))])
    out = Judge(llm).claim_support(ITEMS)
    assert out == {"a-long-hash-1": "SUPPORTED", "b-long-hash-2": "UNSUPPORTED"}
    assert llm.metas[0] == {"prompt": "judge_claim_support.v1"} and "<<<ITEM id=k1" in llm.prompts[0] and "a-long-hash-1" not in llm.prompts[0] and "untrusted" in llm.prompts[0].lower()


def test_judge_retries_when_an_id_is_missing_or_unknown_and_gives_up_after_the_bound():
    llm = ScriptedLLM([labels(("k1", "SUPPORTED")), labels(("k1", "SUPPORTED"), ("k2", "SUPPORTED"))])
    assert Judge(llm).claim_support(ITEMS) == {"a-long-hash-1": "SUPPORTED", "b-long-hash-2": "SUPPORTED"} and llm.calls == 2
    with pytest.raises(MalformedLLMOutput):
        Judge(ScriptedLLM([labels(("k1", "SUPPORTED"), ("k9", "SUPPORTED"))] * 3)).claim_support(ITEMS)


def test_large_batches_are_split_and_empty_input_makes_no_call():
    items = [{"id": f"i{n}", "claim": "c", "quote": "q", "source": "s"} for n in range(25)]
    llm = ScriptedLLM([labels(*[(f"k{n}", "SUPPORTED") for n in range(1, sz + 1)]) for sz in (10, 10, 5)])   # short ids restart in each batch
    assert len(Judge(llm, batch_size=10).claim_support(items)) == 25 and llm.calls == 3
    empty = ScriptedLLM([])
    assert Judge(empty).claim_support([]) == {} and empty.calls == 0


def test_source_relevance_scores_are_validated_to_0_1_2():
    llm = ScriptedLLM([ratings(("r1", 2), ("r2", 0))])
    srcs = [{"id": "s1", "title": "T1", "snippet": "text"}, {"id": "s2", "title": "T2", "snippet": "other"}]
    assert Judge(llm).source_relevance("what is x", srcs) == {"s1": 2, "s2": 0}
    with pytest.raises(MalformedLLMOutput):
        Judge(ScriptedLLM([ratings(("r1", 7), ("r2", 0))] * 3)).source_relevance("q", srcs)


def test_agreement_reports_accuracy_and_cohens_kappa():
    gold = {"1": "SUPPORTED", "2": "SUPPORTED", "3": "UNSUPPORTED", "4": "UNSUPPORTED"}
    perfect = agreement(dict(gold), gold)
    assert perfect["n"] == 4 and perfect["accuracy"] == 1.0 and perfect["kappa"] == 1.0
    flipped = agreement({k: ("UNSUPPORTED" if v == "SUPPORTED" else "SUPPORTED") for k, v in gold.items()}, gold)
    assert flipped["accuracy"] == 0.0 and flipped["kappa"] == -1.0
    half = agreement({"1": "SUPPORTED", "2": "UNSUPPORTED", "3": "UNSUPPORTED", "4": "SUPPORTED"}, gold)
    assert half["accuracy"] == 0.5 and abs(half["kappa"]) < 1e-9
    assert agreement({}, gold)["n"] == 0 and agreement({"1": "SUPPORTED"}, {"9": "SUPPORTED"})["n"] == 0


def test_agreement_with_a_single_class_has_no_kappa_but_reports_accuracy():
    r = agreement({"1": "SUPPORTED", "2": "SUPPORTED"}, {"1": "SUPPORTED", "2": "SUPPORTED"})
    assert r["accuracy"] == 1.0 and r["kappa"] is None
