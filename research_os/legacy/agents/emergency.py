"""Emergency agent: fixed safety message. No tools, no diagnosis, no LLM."""
from __future__ import annotations

from research_os.legacy.agents.base import AgentResult

MESSAGE = ("This sounds like it could be an emergency. Please call your local emergency services "
           "right now (for example 112, 911 or 999) or go to the nearest emergency room. "
           "If someone is with you, ask them to stay with you. I can't diagnose or treat emergencies.")


class EmergencyAgent:
    """Always returns the same safety message."""

    name = "emergency_agent"

    def run(self, text: str, toolbox, llm, intent: str = "EMERGENCY") -> AgentResult:
        return {"agent": self.name, "tool_calls": [], "response": MESSAGE}
