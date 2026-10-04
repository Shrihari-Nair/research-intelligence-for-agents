"""Common agent plumbing: result shape and optional LLM phrasing."""
from __future__ import annotations

from typing import TypedDict


class AgentResult(TypedDict):
    """Uniform agent output."""

    agent: str
    tool_calls: list[dict]
    response: str


def phrase(llm, draft: str) -> str:
    """Let the LLM polish `draft`; fall back to the draft on any problem or if no LLM."""
    if llm is None:
        return draft
    try:
        out = llm.generate("Rewrite this clinic assistant reply to be brief and friendly, keeping "
                           f"every fact unchanged.\nDraft: {draft}")
        return out.strip() or draft
    except Exception:
        return draft
