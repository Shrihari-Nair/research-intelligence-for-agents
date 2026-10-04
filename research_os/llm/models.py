"""Role-based model configuration (configs/models.yaml)."""
from __future__ import annotations

from research_os.common import load_config

ROLES = ("router_fallback", "supervisor", "planner", "evidence", "verifier_judge", "synthesis", "data_generation", "eval_judge")
_PATH = "configs/models.yaml"


def model_for_role(role: str, path: str = _PATH) -> str:
    """Model name configured for `role`; KeyError if the role is unknown."""
    roles = load_config(path)["roles"]
    if role not in roles:
        raise KeyError(f"unknown model role {role!r}; known: {sorted(roles)}")
    return roles[role]


def provider_settings(path: str = _PATH) -> dict:
    """Provider-level settings: provider, api_key_env (env var name only), timeout_s."""
    cfg = load_config(path)
    return {"provider": cfg["provider"], "api_key_env": cfg["api_key_env"], "timeout_s": cfg["timeout_s"]}
