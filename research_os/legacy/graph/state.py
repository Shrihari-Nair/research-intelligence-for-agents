"""LangGraph state."""
from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """Flows through the graph: input text -> routing record -> agent output -> response."""

    text: str
    routing: dict
    agent_name: str
    tool_calls: list
    response: str
