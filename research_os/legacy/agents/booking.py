"""Booking agent: book / reschedule / cancel."""
from __future__ import annotations

from research_os.legacy.agents.base import AgentResult, phrase
from research_os.legacy.agents.parsing import (
    DEFAULT_PATIENT,
    extract_day,
    extract_doctor,
    extract_time,
    extract_unknown_doctor,
    extract_unsupported_day,
)


def _fmt(a: dict) -> str:
    return f"{a['doctor']} on {a['day']} at {a['time']}"


class BookingAgent:
    """Handles BOOK_/RESCHEDULE_/CANCEL_APPOINTMENT with the mock toolbox."""

    name = "booking_agent"

    def run(self, text: str, toolbox, llm, intent: str = "BOOK_APPOINTMENT") -> AgentResult:
        calls: list[dict] = []

        def call(name: str, **kw):
            calls.append({"name": name, "args": kw})
            return getattr(toolbox, name)(**kw)

        doctor, day, time = extract_doctor(text), extract_day(text), extract_time(text)
        bad_day = extract_unsupported_day(text)
        if intent == "CANCEL_APPOINTMENT":
            r = call("cancel_appointment", patient_id=DEFAULT_PATIENT)
            draft = (f"Your appointment with {_fmt(r['cancelled'])} has been cancelled."
                     if r.get("ok") else "I couldn't find an appointment to cancel; you have no appointment on file.")
        elif bad_day:
            draft = f"Sorry, we're closed on {bad_day}. We're open Monday to Saturday; which day would you like?"
        elif intent == "RESCHEDULE_APPOINTMENT":
            r = call("reschedule_appointment", patient_id=DEFAULT_PATIENT, day=day, time=time)
            if r.get("ok"):
                draft = f"Done. Your appointment is now {_fmt(r['appointment'])}."
            else:
                wanted = f" to {time}" if time else ""
                draft = f"I couldn't reschedule{wanted}: {r['error']}."
        elif (unknown := extract_unknown_doctor(text)):
            draft = f"Sorry, I don't see {unknown} on our roster. Would you like a different doctor?"
        else:
            slots = call("check_slots", doctor=doctor, day=day)["slots"]
            exact = [s for s in slots if s["time"] == time] if time else slots
            if not slots:
                draft = "Sorry, I couldn't find an open slot for that request. Would another day work?"
            elif not exact:
                who = f" with {doctor}" if doctor else ""
                options = ", ".join(sorted({s["time"] for s in slots})[:4])
                draft = f"{time} is not available{who}. Open times that day: {options}. Which would you like?"
            else:
                slots = exact
                s = slots[0]
                r = call("book_appointment", patient_id=DEFAULT_PATIENT, doctor=s["doctor"],
                         day=s["day"], time=s["time"])
                draft = (f"You're booked with {_fmt(r['appointment'])}." if r.get("ok")
                         else f"I couldn't book that: {r['error']}.")
        return {"agent": self.name, "tool_calls": calls, "response": phrase(llm, draft)}
