"""Gemini-backed STT and TTS adapters (use the same GOOGLE_API_KEY as the LLM; no extra service).

STT sends a WAV to a Gemini model and asks for a verbatim transcript; TTS asks a Gemini TTS model for raw PCM16 (24 kHz mono).
Both take an injected `client` (anything with `.models.generate_content`) so they can be tested without the network.
"""
from __future__ import annotations

import os
from typing import Callable

from research_os.common import load_config

_STT_PROMPT = "Transcribe this audio exactly. Output only the transcript, nothing else."


def _client(client=None):
    if client is not None:
        return client
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


class GeminiSTT:
    """Speech (WAV bytes) -> transcript. A str is passed through unchanged (text path)."""

    def __init__(self, client=None, model: str | None = None):
        self.client = _client(client)
        self.model = model or load_config("configs/voice.yaml")["stt_model"]

    def transcribe(self, audio: str | bytes) -> str:
        if isinstance(audio, str):
            return audio
        if not audio:
            raise ValueError("empty audio")
        from google.genai import types
        r = self.client.models.generate_content(model=self.model, contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav"), _STT_PROMPT])
        return (r.text or "").strip()


class GeminiTTS:
    """Reply text -> PCM16 mono audio at `sample_rate`; `speak` hands the audio to `sink` (e.g. a LiveKit audio source)."""

    sample_rate = 24000

    def __init__(self, client=None, model: str | None = None, voice: str | None = None, sink: Callable[[bytes], None] | None = None):
        cfg = load_config("configs/voice.yaml")
        self.client, self.model, self.voice, self.sink = _client(client), model or cfg["tts_model"], voice or cfg["voice"], sink

    def synthesize(self, text: str) -> bytes:
        from google.genai import types
        cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))))
        r = self.client.models.generate_content(model=self.model, contents=text, config=cfg)
        return r.candidates[0].content.parts[0].inline_data.data

    def speak(self, text: str) -> None:
        pcm = self.synthesize(text)
        if self.sink is not None:
            self.sink(pcm)
