"""Deterministic MOCK tools over in-memory fake data. Errors are returned, never raised."""
from __future__ import annotations

DOCTORS = ["Dr Sharma", "Dr Patel", "Dr Rao", "Dr Iyer", "Dr Mehta", "Dr Khan"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "tomorrow"]
TIMES = ["09:00", "10:30", "14:00", "16:30"]

_CANNED_PUBMED = [
    {"title": "Recent advances in the management of {q}", "source": "PubMed (canned)", "year": 2024},
    {"title": "A systematic review of treatment outcomes: {q}", "source": "PubMed (canned)", "year": 2023},
]
_CANNED_WEB = [{"title": "Patient overview: {q}", "source": "Web (canned)"}]


class ToolBox:
    """Fake clinic backend. Patient P001 starts with an appointment, prescription and follow-up."""

    def __init__(self, seed: int = 42, use_pubmed: bool = False):
        self.use_pubmed = use_pubmed  # real PubMed is intentionally not implemented by default
        self.free: set[tuple[str, str, str]] = {(d, day, t) for d in DOCTORS for day in DAYS for t in TIMES}
        self.patients = {"P001": {"name": "Asha Verma"}, "P002": {"name": "Ravi Kumar"},
                         "P003": {"name": "Meera Das"}}
        self.appointments: dict[str, list[dict]] = {}
        self.free.discard(("Dr Sharma", "wednesday", "10:30"))
        self.appointments["P001"] = [{"doctor": "Dr Sharma", "day": "wednesday", "time": "10:30"}]
        self.prescriptions = {"P001": [
            {"drug": "metformin", "dose": "500 mg", "schedule": "twice a day after meals"},
            {"drug": "atorvastatin", "dose": "10 mg", "schedule": "once at night"}]}
        self.followups = {"P001": "Return for a review in 2 weeks with fasting blood sugar results."}

    def _unknown(self, pid: str) -> dict | None:
        return None if pid in self.patients else {"error": f"unknown patient {pid}"}

    def get_appointment(self, patient_id: str) -> dict | None:
        """The patient's earliest-booked appointment, if any."""
        appts = self.appointments.get(patient_id)
        return appts[0] if appts else None

    def check_slots(self, doctor: str | None = None, day: str | None = None) -> dict:
        """List free slots, optionally filtered by doctor and day."""
        slots = [{"doctor": d, "day": dy, "time": t} for d, dy, t in sorted(self.free, key=lambda s: (
            DAYS.index(s[1]), s[2], s[0])) if (not doctor or d == doctor) and (not day or dy == day)]
        return {"slots": slots}

    def book_appointment(self, patient_id: str, doctor: str, day: str, time: str) -> dict:
        """Book a free slot for a patient (patients may hold several appointments)."""
        if (e := self._unknown(patient_id)):
            return e
        if (doctor, day, time) not in self.free:
            return {"error": f"slot {doctor} {day} {time} is not available"}
        self.free.discard((doctor, day, time))
        appt = {"doctor": doctor, "day": day, "time": time}
        self.appointments.setdefault(patient_id, []).append(appt)
        return {"ok": True, "appointment": appt}

    def reschedule_appointment(self, patient_id: str, day: str | None = None, time: str | None = None) -> dict:
        """Move the patient's appointment to the first free slot (same doctor) on `day`/`time`."""
        if (e := self._unknown(patient_id)):
            return e
        cur = self.get_appointment(patient_id)
        if not cur:
            return {"error": "no appointment to reschedule"}
        options = [s for s in self.check_slots(cur["doctor"], day)["slots"] if not time or s["time"] == time]
        if not options:
            return {"error": f"no free slot with {cur['doctor']} for that request"}
        new = options[0]
        self.free.add((cur["doctor"], cur["day"], cur["time"]))
        self.free.discard((new["doctor"], new["day"], new["time"]))
        self.appointments[patient_id][0] = new
        return {"ok": True, "appointment": new, "previous": cur}

    def cancel_appointment(self, patient_id: str) -> dict:
        """Cancel and free the patient's appointment."""
        if (e := self._unknown(patient_id)):
            return e
        appts = self.appointments.get(patient_id)
        if not appts:
            return {"error": "no appointment to cancel"}
        cur = appts.pop(0)
        self.free.add((cur["doctor"], cur["day"], cur["time"]))
        return {"ok": True, "cancelled": cur}

    def get_prescription(self, patient_id: str) -> dict:
        """Current medications for a patient."""
        if (e := self._unknown(patient_id)):
            return e
        meds = self.prescriptions.get(patient_id, [])
        return {"medications": meds} if meds else {"error": "no prescriptions on file"}

    def get_followup(self, patient_id: str) -> dict:
        """Follow-up instructions for a patient."""
        if (e := self._unknown(patient_id)):
            return e
        fu = self.followups.get(patient_id)
        return {"followup": fu} if fu else {"error": "no follow-up on file"}

    def search_pubmed(self, query: str) -> dict:
        """Canned PubMed-style results (offline, deterministic)."""
        return {"results": [{**r, "title": r["title"].format(q=query)} for r in _CANNED_PUBMED]}

    def search_web(self, query: str) -> dict:
        """Canned web results (offline, deterministic)."""
        return {"results": [{**r, "title": r["title"].format(q=query)} for r in _CANNED_WEB]}
