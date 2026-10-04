"""Tiny regex extraction of doctor / day / time from an utterance."""
from __future__ import annotations

import re

from research_os.legacy.agents.tools import DAYS, DOCTORS

DEFAULT_PATIENT = "P001"
UNSUPPORTED_DAYS = ("sunday",)
_DOCTOR_RE = re.compile(r"\b(?:dr|doctor)\.?\s+([a-z]+)")


def extract_doctor(text: str) -> str | None:
    """Return a rostered 'Dr X' named in the text ('Dr X', 'Dr. X' or 'doctor X'), else None."""
    roster = {d.split()[1].lower(): d for d in DOCTORS}
    for m in _DOCTOR_RE.finditer(text.lower()):
        if m.group(1) in roster:
            return roster[m.group(1)]
    return None


def extract_unknown_doctor(text: str) -> str | None:
    """Return 'Dr X' if the text names a doctor who is not on the roster, else None."""
    m = _DOCTOR_RE.search(text.lower())
    return f"Dr {m.group(1).title()}" if m and extract_doctor(text) is None else None


def extract_day(text: str) -> str | None:
    """Return a recognised day word (weekday or 'tomorrow'), else None."""
    low = text.lower()
    return next((d for d in DAYS if re.search(rf"\b{d}\b", low)), None)


def extract_unsupported_day(text: str) -> str | None:
    """Return a day word we cannot book (e.g. 'sunday'), else None."""
    low = text.lower()
    return next((d for d in UNSUPPORTED_DAYS if re.search(rf"\b{d}\b", low)), None)


def extract_time(text: str) -> str | None:
    """Return 'HH:MM' for the first real clock time ('3 pm', '10:30'), ignoring other numbers.

    A bare 'H:MM' with H in 1-6 and no am/pm is read as afternoon (clinics are not open at 2 am).
    """
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text.lower()):
        if not (m.group(2) or m.group(3)):
            continue
        h, mi, ap = int(m.group(1)), m.group(2) or "00", m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        elif ap is None and 1 <= h <= 6:
            h += 12
        if h > 23 or int(mi) > 59:
            continue
        return f"{h:02d}:{mi}"
    return None
