import pandas as pd
import pytest

from research_os.common import intents_from_cfg
from research_os.router.data.load import load_split


def test_load_split_reads_jsonl_with_validation_alias(tmp_path):
    (tmp_path / "validation.jsonl").write_text('{"text": "a b", "intent": "X"}\n{"text": "c d", "intent": "Y"}\n')
    df = load_split("val", str(tmp_path))
    assert list(df.columns) == ["text", "intent"] and len(df) == 2


def test_load_split_reads_csv(tmp_path):
    pd.DataFrame({"text": ["a b"], "intent": ["X"]}).to_csv(tmp_path / "train.csv", index=False)
    assert len(load_split("train", str(tmp_path))) == 1


def test_load_split_missing_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_split("train", str(tmp_path))


def test_intents_from_cfg_uses_configured_path(tmp_path):
    f = tmp_path / "i.yaml"
    f.write_text("intents: [A, B]\n")
    assert intents_from_cfg({"paths": {"intents": str(f)}}) == ["A", "B"]
    assert intents_from_cfg({"paths": {}}) == intents_from_cfg({})  # falls back to configs/intents.yaml
