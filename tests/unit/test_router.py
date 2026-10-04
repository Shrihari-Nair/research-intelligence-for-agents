import pytest

from research_os.common import load_intents
from research_os.router.model import IntentRouter


@pytest.fixture(scope="module")
def router(tiny_model_dir):
    return IntentRouter.load(tiny_model_dir)


def test_predict_schema(router):
    out = router.predict("can I move my appointment to friday")
    assert set(out) == {"intent", "confidence", "probs"}
    assert out["intent"] in load_intents()
    assert 0.0 <= out["confidence"] <= 1.0
    assert list(out["probs"]) == load_intents()
    assert sum(out["probs"].values()) == pytest.approx(1.0, abs=1e-4)
    assert out["confidence"] == pytest.approx(max(out["probs"].values()))
    assert out["probs"][out["intent"]] == pytest.approx(out["confidence"])


def test_batch_matches_single(router):
    texts = ["cancel my appointment", "what are your opening hours", "i cant breathe"]
    batch = router.predict_batch(texts)
    for t, b in zip(texts, batch, strict=True):
        s = router.predict(t)
        assert s["intent"] == b["intent"]
        assert s["confidence"] == pytest.approx(b["confidence"], abs=1e-5)


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_empty_text_raises(router, bad):
    with pytest.raises(ValueError):
        router.predict(bad)


def test_reload_gives_identical_predictions(tiny_model_dir):
    a, b = IntentRouter.load(tiny_model_dir), IntentRouter.load(tiny_model_dir)
    t = "when should I follow up"
    assert a.predict(t)["probs"] == pytest.approx(b.predict(t)["probs"])
