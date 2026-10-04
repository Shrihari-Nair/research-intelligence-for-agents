"""Small FAQ responder for GENERAL_QUERY."""
from __future__ import annotations

from research_os.legacy.agents.base import AgentResult, phrase

FAQ = [
    (("hour", "open", "close", "sunday", "holiday"), "We're open Monday to Saturday, 9 am to 6 pm."),
    (("where", "locat", "address", "reach", "transport"),
     "The clinic is at 12 Park Street, near the central metro station."),
    (("insurance", "pay", "cost", "price", "fee", "bill"),
     "We accept most major insurance plans, cards and UPI; a consultation starts at 500."),
    (("service", "provide", "offer", "what do you", "can you help", "what can you"),
     "We offer general consultations, specialist visits, lab tests, vaccinations and a pharmacy."),
    (("parking", "wheelchair", "access"), "Yes, we have parking and step-free wheelchair access."),
    (("telemedicine", "online", "video"), "Yes, video consultations are available by appointment."),
]
DEFAULT = "I can help with appointments, prescriptions, follow-ups and general clinic questions."


class FAQAgent:
    """Keyword FAQ lookup."""

    name = "faq_agent"

    def run(self, text: str, toolbox, llm, intent: str = "GENERAL_QUERY") -> AgentResult:
        low = text.lower()
        draft = next((a for kws, a in FAQ if any(k in low for k in kws)), DEFAULT)
        return {"agent": self.name, "tool_calls": [], "response": phrase(llm, draft)}
