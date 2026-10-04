import random

import numpy as np
import torch

from research_os.common import get_hardware_info, load_config, load_intents, set_seed


def test_intents_are_the_eight_unique_labels():
    intents = load_intents()
    assert len(intents) == 8 and len(set(intents)) == 8
    assert "GENERAL_RESEARCH" in intents


def test_set_seed_is_reproducible():
    set_seed(7)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(7)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_hardware_info_keys():
    info = get_hardware_info()
    for k in ("cpu_model", "cores", "ram_gb", "torch_version", "torch_threads"):
        assert k in info


def test_load_config_reads_yaml():
    assert load_config("configs/training.yaml")["seed"] == 42


def test_load_dotenv_sets_missing_but_never_overrides(tmp_path, monkeypatch):
    from research_os.common import load_dotenv

    f = tmp_path / ".env"
    f.write_text("# comment\nFOO_TEST_A=one\nFOO_TEST_B='two words'\nFOO_TEST_C=three\n\nbad line\n")
    monkeypatch.delenv("FOO_TEST_A", raising=False)
    monkeypatch.delenv("FOO_TEST_B", raising=False)
    monkeypatch.setenv("FOO_TEST_C", "already")
    load_dotenv(str(f))
    import os
    assert os.environ["FOO_TEST_A"] == "one"
    assert os.environ["FOO_TEST_B"] == "two words"
    assert os.environ["FOO_TEST_C"] == "already"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    from research_os.common import load_dotenv

    load_dotenv(str(tmp_path / "nope.env"))
