"""Prescription / follow-up agent."""
from __future__ import annotations

from research_os.legacy.agents.base import AgentResult, phrase
from research_os.legacy.agents.parsing import DEFAULT_PATIENT


class PrescriptionAgent:
    """Answers PRESCRIPTION_QUERY and FOLLOWUP_QUERY from the mock records."""

    name = "prescription_agent"

    def run(self, text: str, toolbox, llm, intent: str = "PRESCRIPTION_QUERY") -> AgentResult:
        if intent == "FOLLOWUP_QUERY":
            calls = [{"name": "get_followup", "args": {"patient_id": DEFAULT_PATIENT}}]
            r = toolbox.get_followup(DEFAULT_PATIENT)
            draft = r["followup"] if "followup" in r else f"I couldn't find follow-up details: {r['error']}."
        else:
            calls = [{"name": "get_prescription", "args": {"patient_id": DEFAULT_PATIENT}}]
            r = toolbox.get_prescription(DEFAULT_PATIENT)
            if "medications" in r:
                meds = "; ".join(f"{m['drug']} {m['dose']}, {m['schedule']}" for m in r["medications"])
                draft = f"Your current prescription: {meds}."
            else:
                draft = f"I couldn't find a prescription: {r['error']}."
        return {"agent": self.name, "tool_calls": calls, "response": phrase(llm, draft)}
