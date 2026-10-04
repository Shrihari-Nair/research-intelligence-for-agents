"""STT/TTS adapter interfaces. Defaults are text no-ops; LiveKit etc. live in requirements-voice.txt."""
from __future__ import annotations

from typing import Protocol

from research_os.legacy.graph.workflow import run_text


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


def voice_turn(stt: STTAdapter, graph, tts: TTSAdapter, audio: str | bytes):
    """mic -> STT -> router -> LangGraph -> tool -> response -> TTS. Returns None on empty speech."""
    transcript = stt.transcribe(audio).strip()
    if not transcript:
        return None
    state = run_text(graph, transcript)
    tts.speak(state["response"])
    return state
