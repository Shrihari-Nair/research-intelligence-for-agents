import json

import pytest

from research_os.llm.mock_client import ScriptedLLM
from research_os.llm.structured import MalformedLLMOutput
from research_os.router.data.llm_generate import STYLES, generate_rows


def batch(*texts):
    return json.dumps({"utterances": list(texts)})


def test_generate_rows_labels_and_metadata():
    llm = ScriptedLLM([batch("compare a and b please", "how does x work")] * len(STYLES))
    rows = generate_rows(llm, "TECHNOLOGY_COMPARISON", n_per_style=2, model="m")
    assert len(rows) == 2 * len(STYLES)
    assert {r["intent"] for r in rows} == {"TECHNOLOGY_COMPARISON"}
    assert {r["style"] for r in rows} == set(STYLES)
    assert all(r["model"] == "m" and r["prompt_version"] for r in rows)


def test_prompt_names_intent_definition_and_style():
    llm = ScriptedLLM([batch("a b c")] * len(STYLES))
    generate_rows(llm, "RESEARCH_MONITORING", n_per_style=1, model="m")
    assert "RESEARCH_MONITORING" in llm.prompts[0] and "ongoing" in llm.prompts[0].lower()
    assert STYLES[0] in llm.prompts[0] and STYLES[1] in llm.prompts[1]


def test_short_or_empty_items_are_dropped():
    llm = ScriptedLLM([batch("ok", "", "a perfectly fine utterance here")] * len(STYLES))
    rows = generate_rows(llm, "FACT_LOOKUP", n_per_style=3, model="m")
    assert all(len(r["text"].split()) >= 3 for r in rows) and len(rows) == len(STYLES)


def test_malformed_output_propagates_after_bounded_retries():
    llm = ScriptedLLM(["nope"] * 10)
    with pytest.raises(MalformedLLMOutput):
        generate_rows(llm, "FACT_LOOKUP", n_per_style=2, model="m")
    assert llm.calls == 3
