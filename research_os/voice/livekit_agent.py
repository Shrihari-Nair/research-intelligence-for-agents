"""LiveKit voice agent: caller audio -> STT -> `voice_turn` (normal research run) -> TTS audio published back into the room.

Uses the `livekit` rtc SDK only (optional extra, requirements-voice.txt); VAD is the energy-based `SilenceSegmenter`, so no model download.
The blocking part of a turn (STT, the research run, TTS) runs in a worker thread so the room's audio keeps flowing.
"""
from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from research_os.voice.adapters import voice_turn
from research_os.voice.audio import SilenceSegmenter, pcm_to_wav

log = logging.getLogger(__name__)
STT_RATE = 16000


class LiveKitUnavailable(RuntimeError):
    """The optional LiveKit dependency group is not installed."""


def require_livekit() -> None:
    """Raise a clear error unless the optional LiveKit packages are importable."""
    try:
        import livekit  # noqa: F401
    except ImportError as e:
        raise LiveKitUnavailable("LiveKit is not installed. Install the optional voice extra: pip install -r requirements-voice.txt") from e


def make_token(identity: str, room: str, api_key: str | None = None, api_secret: str | None = None) -> str:
    """Join token for `room`. Keys default to LIVEKIT_API_KEY / LIVEKIT_API_SECRET (never hard-coded)."""
    require_livekit()
    from livekit import api
    key, secret = api_key or os.environ["LIVEKIT_API_KEY"], api_secret or os.environ["LIVEKIT_API_SECRET"]
    return api.AccessToken(key, secret).with_identity(identity).with_grants(api.VideoGrants(room_join=True, room=room)).to_jwt()


class VoiceAgent:
    """Joins a room, answers each spoken request with a short spoken summary of the finished research run."""

    def __init__(self, service, stt, tts, collected_pcm: list[bytes], cfg: dict, *, base_url: str = "http://localhost:8000", identity: str = "research-agent"):
        self.service, self.stt, self.tts, self._pcm, self.cfg, self.base_url, self.identity = service, stt, tts, collected_pcm, cfg, base_url, identity
        self.turns: list[dict] = []
        self.room = None
        self._source = None
        self._tasks: set[asyncio.Task] = set()

    async def connect(self, url: str, token: str) -> None:
        require_livekit()
        from livekit import rtc
        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track)
        await self.room.connect(url, token)
        self._source = rtc.AudioSource(self.tts.sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", self._source)
        await self.room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self.room is not None:
            await self.room.disconnect()

    def _on_track(self, track, publication, participant) -> None:
        from livekit import rtc
        if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity != self.identity:
            t = asyncio.get_running_loop().create_task(self._listen(track, participant.identity))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def _listen(self, track, who: str) -> None:
        from livekit import rtc
        lk = self.cfg["livekit"]
        seg = SilenceSegmenter(STT_RATE, lk["energy_threshold"], lk["silence_ms"], lk["min_speech_ms"])
        async for ev in rtc.AudioStream(track, sample_rate=STT_RATE, num_channels=1):
            utt = seg.push(np.frombuffer(bytes(ev.frame.data), dtype=np.int16))
            if utt is not None:
                await self._answer(utt, who)

    async def _answer(self, pcm: bytes, who: str) -> None:
        self._pcm.clear()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: voice_turn(self.stt, self.service, self.tts, pcm_to_wav(pcm, STT_RATE), base_url=self.base_url))
        if result is None:
            return
        self.turns.append({"who": who, **result})
        log.info("voice turn %s -> %s", result["status"], result["spoken"][:80])
        await self._play(b"".join(self._pcm))

    async def _play(self, pcm: bytes) -> None:
        from livekit import rtc
        rate, step = self.tts.sample_rate, self.tts.sample_rate // 100          # 10 ms frames
        samples = np.frombuffer(pcm, dtype=np.int16)
        for i in range(0, len(samples), step):
            chunk = samples[i:i + step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            await self._source.capture_frame(rtc.AudioFrame(chunk.tobytes(), rate, 1, step))
        await self._source.wait_for_playout()
