"""Small, dependency-light audio helpers for the voice path (PCM16 mono)."""
from __future__ import annotations

import io
import wave

import numpy as np


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian PCM16 mono samples in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class SilenceSegmenter:
    """Energy-based utterance detector: push PCM16 frames; returns the utterance's PCM bytes once speech is followed by `silence_ms` of quiet.

    Speech shorter than `min_speech_ms` is treated as noise and dropped. `flush()` returns speech still pending (e.g. the stream ended).
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 500.0, silence_ms: int = 700, min_speech_ms: int = 250):
        self.rate, self.threshold, self.silence_ms, self.min_speech_ms = sample_rate, threshold, silence_ms, min_speech_ms
        self._chunks: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._quiet_ms = 0.0

    def _reset(self) -> None:
        self._chunks, self._speech_ms, self._quiet_ms = [], 0.0, 0.0

    def push(self, samples: np.ndarray) -> bytes | None:
        ms = len(samples) * 1000 / self.rate
        loud = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) >= self.threshold if len(samples) else False
        if loud:
            self._chunks.append(samples)
            self._speech_ms += ms
            self._quiet_ms = 0.0
            return None
        if not self._chunks:
            return None
        self._chunks.append(samples)
        self._quiet_ms += ms
        if self._quiet_ms < self.silence_ms:
            return None
        return self.flush()

    def flush(self) -> bytes | None:
        out = np.concatenate(self._chunks).tobytes() if self._chunks and self._speech_ms >= self.min_speech_ms else None
        self._reset()
        return out
