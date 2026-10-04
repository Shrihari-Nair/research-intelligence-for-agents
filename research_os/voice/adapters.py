"""Voice layer (optional, M24): STT/TTS adapter interfaces (text no-ops by default) and a voice turn around the normal run pipeline.

Voice never changes the core pipeline: the transcript is submitted as an ordinary request; the reply is a short spoken summary of the
finished ResearchReport plus a link to the full report.
"""
from __future__ import annotations

import re
import time
from typing import Protocol

_SENT = re.compile(r"(?<=[.!?])\s+")


class STTAdapter(Protocol):
    """Speech (or text) in -> transcript."""

    def transcribe(self, audio: str | bytes) -> str: ...


class TTSAdapter(Protocol):
    """Reply text -> speech out."""

    def speak(self, text: str) -> None: ...


class TextSTT:
    """No-op STT: the 'audio' already is the transcript."""

    def transcribe(self, audio: str | bytes) -> str:
        return audio.decode() if isinstance(audio, bytes) else audio


class TextTTS:
    """No-op TTS: records what would have been spoken."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


def spoken_summary(text: str, max_sentences: int = 2, max_words: int = 45) -> str:
    """First sentences of a summary, stripped of markdown and citation markers, capped in words (for speech)."""
    t = re.sub(r"\[\d+\]", "", text or "")
    t = re.sub(r"[*_`#>]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    out = " ".join(_SENT.split(t)[:max_sentences])
    words = out.split()
    return out if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "…"


def voice_turn(stt: STTAdapter, service, tts: TTSAdapter, audio: str | bytes, *, base_url: str = "http://localhost:8000",
               timeout_s: float = 60.0, max_sentences: int = 2, max_words: int = 45) -> dict | None:
    """mic -> STT -> router -> Research OS -> TTS. Returns None for an empty transcript, else {run_id, status, spoken, report_url}."""
    transcript = stt.transcribe(audio).strip()
    if not transcript:
        return None
    rec = service.submit(transcript)
    t0 = time.time()
    while not rec.done and rec.status.value != "AWAITING_APPROVAL" and time.time() - t0 < timeout_s:
        time.sleep(0.05)
    url = f"{base_url}/runs/{rec.run_id}/report?format=pdf"
    status = rec.status.value
    if status == "AWAITING_APPROVAL":
        spoken = "This research needs your approval first. Please open the app to approve it."
    elif not rec.done:
        spoken = "I'm still working on that. The full report will be in your run history when it's ready."
    elif status == "FAILED":
        spoken = "Sorry, I couldn't complete that research. Please try rephrasing it."
    elif status == "CANCELLED":
        spoken = "That request was cancelled."
    else:
        summary = spoken_summary(rec.basic_report().get("executive_summary", ""), max_sentences, max_words)
        partial = " This result is partial." if status == "PARTIAL" else ""
        spoken = f"{summary}{partial} The full report with sources is in your run history.".strip()
    tts.speak(spoken)
    return {"run_id": rec.run_id, "status": status, "spoken": spoken, "report_url": url}
