import io
import wave

import numpy as np
import pytest

from research_os.voice.audio import SilenceSegmenter, pcm_to_wav
from research_os.voice.gemini_audio import GeminiSTT, GeminiTTS


def frame(amp, ms=20, rate=16000):
    return (np.full(rate * ms // 1000, amp, dtype=np.int16))


def test_pcm_to_wav_is_a_readable_mono_pcm16_wav():
    wav = pcm_to_wav(frame(1000, 100).tobytes(), 16000)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 1600)


def test_segmenter_emits_one_utterance_after_speech_then_silence():
    seg = SilenceSegmenter(sample_rate=16000, threshold=500, silence_ms=300, min_speech_ms=100)
    out = []
    for f in [frame(0)] * 5 + [frame(3000)] * 15 + [frame(0)] * 20:        # 100 ms quiet, 300 ms speech, 400 ms quiet
        done = seg.push(f)
        if done is not None:
            out.append(done)
    assert len(out) == 1
    assert 300 <= len(out[0]) / 32 <= 700                                  # ms of PCM16 audio: the speech plus the trailing silence


def test_segmenter_ignores_a_short_blip_and_flushes_speech_cut_off_by_the_end():
    seg = SilenceSegmenter(sample_rate=16000, threshold=500, silence_ms=200, min_speech_ms=200)
    assert [seg.push(f) for f in [frame(3000)] * 2 + [frame(0)] * 20] == [None] * 22      # 40 ms blip: no utterance
    seg2 = SilenceSegmenter(sample_rate=16000, threshold=500, silence_ms=200, min_speech_ms=100)
    for f in [frame(3000)] * 15:
        assert seg2.push(f) is None
    assert seg2.flush() is not None and seg2.flush() is None


class FakeModels:
    def __init__(self, text=None, pcm=None):
        self.calls, self.text, self.pcm = [], text, pcm

    def generate_content(self, *, model, contents, config=None):
        self.calls.append((model, contents, config))
        from types import SimpleNamespace as N
        if self.pcm is not None:
            part = N(inline_data=N(data=self.pcm, mime_type="audio/L16;codec=pcm;rate=24000"))
            return N(text=None, candidates=[N(content=N(parts=[part]))])
        return N(text=self.text, candidates=[])


class FakeClient:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


def test_gemini_stt_sends_wav_bytes_and_returns_the_stripped_transcript():
    client = FakeClient(text="  what license does langgraph use \n")
    stt = GeminiSTT(client=client, model="m")
    assert stt.transcribe(pcm_to_wav(b"\x00\x00" * 160, 16000)) == "what license does langgraph use"
    model, contents, _ = client.models.calls[0]
    assert model == "m" and any(getattr(c, "inline_data", None) is not None for c in contents)


def test_gemini_stt_passes_text_through_and_rejects_empty_audio():
    assert GeminiSTT(client=FakeClient(text="x"), model="m").transcribe("already text") == "already text"
    with pytest.raises(ValueError):
        GeminiSTT(client=FakeClient(text="x"), model="m").transcribe(b"")


def test_gemini_tts_returns_pcm_and_forwards_it_to_the_sink():
    got = []
    tts = GeminiTTS(client=FakeClient(pcm=b"\x01\x00" * 100), model="t", voice="Kore", sink=got.append)
    assert tts.synthesize("hello") == b"\x01\x00" * 100
    tts.speak("hello")
    assert got == [b"\x01\x00" * 100] and tts.sample_rate == 24000
