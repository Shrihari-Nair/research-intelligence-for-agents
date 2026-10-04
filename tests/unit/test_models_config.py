import pytest

from research_os.llm.models import ROLES, model_for_role, provider_settings


def test_every_role_has_a_model_name():
    for role in ROLES:
        assert model_for_role(role)


def test_eval_judge_differs_from_verifier_judge():
    assert model_for_role("eval_judge") != model_for_role("verifier_judge")


def test_unknown_role_raises():
    with pytest.raises(KeyError, match="nope"):
        model_for_role("nope")


def test_provider_settings_name_env_var_not_secret():
    s = provider_settings()
    assert s["api_key_env"] == "GOOGLE_API_KEY" and "key" not in {k for k in s if k != "api_key_env"}
