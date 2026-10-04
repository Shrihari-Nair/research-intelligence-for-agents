"""Shared agent plumbing."""
from __future__ import annotations

from research_os.common import load_config
from research_os.contracts import FailureType


class AgentFailure(Exception):
    """A classified agent failure (the executor reads `failure_type`)."""

    def __init__(self, failure_type: FailureType, message: str):
        super().__init__(message)
        self.failure_type = failure_type


def agent_cfg(path: str = "configs/agents.yaml") -> dict:
    """Agent settings from config."""
    return load_config(path)


def merged(inputs: dict[str, dict], key: str) -> list:
    """Concatenate list-valued `key` across all dependency outputs, preserving order and dropping duplicates."""
    out: list = []
    for dep in inputs.values():
        for item in dep.get(key, []):
            if item not in out:
                out.append(item)
    return out
