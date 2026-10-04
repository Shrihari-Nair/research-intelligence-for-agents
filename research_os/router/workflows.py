"""intent -> workflow lookup from configs/workflows.yaml."""
from __future__ import annotations

from research_os.common import load_config


def workflow_map(path: str = "configs/workflows.yaml") -> dict[str, str]:
    """{intent: workflow_name}; raises if two workflows claim the same intent."""
    out: dict[str, str] = {}
    for name, spec in load_config(path)["workflows"].items():
        if spec["intent"] in out:
            raise ValueError(f"intent {spec['intent']} mapped to two workflows: {out[spec['intent']]}, {name}")
        out[spec["intent"]] = name
    return out


def workflow_for_intent(intent: str, path: str = "configs/workflows.yaml") -> str:
    """Workflow name for an intent (KeyError if unknown)."""
    m = workflow_map(path)
    if intent not in m:
        raise KeyError(f"no workflow configured for intent {intent!r}")
    return m[intent]
