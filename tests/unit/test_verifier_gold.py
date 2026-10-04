import json
from collections import Counter

from evaluation.verifier_eval import evaluate_pairs, load_gold
from research_os.llm.mock_client import ScriptedLLM


def test_gold_set_is_wellformed_balanced_and_quotes_are_verbatim():
    gold = load_gold()
    assert len(gold) == 30 and len({g["id"] for g in gold}) == 30
    assert Counter(g["expected"] for g in gold) == {"SUPPORTED": 10, "UNSUPPORTED": 10, "CONTRADICTED": 10}
    assert all(g["evidence_quote"] in g["source_text"] for g in gold)


def test_evaluate_pairs_reports_accuracy_confusion_and_deterministic_breakdown():
    gold = load_gold()
    # a perfect scripted judge for everything the deterministic layer lets through
    llm = ScriptedLLM([])

    def perfect(prompt, **kw):
        import re
        by_claim = {g["claim"]: g["expected"] for g in gold}
        pairs = re.findall(r"<<<CLAIM id=(\S+)\nClaim: (.*)\n", prompt)          # short ids (c1..cN) + the claim text
        return json.dumps({"judgements": [{"claim_id": i, "verdict": by_claim[text], "rationale": "gold"} for i, text in pairs]})
    from research_os.llm.client import LLMResponse
    llm.complete = lambda prompt, **kw: LLMResponse(text=perfect(prompt))
    res = evaluate_pairs(gold, llm)
    assert res["n"] == 30 and 0.9 <= res["accuracy"] <= 1.0
    assert set(res["confusion"]) == {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
    assert res["deterministic"]["false_rejections_of_supported"] == 0
    assert res["deterministic"]["rejected"] >= 8     # overreach pairs are caught without the judge
