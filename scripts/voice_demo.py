"""End-to-end voice demo against a LIVE LiveKit server (start one: see docs/VOICE.md).

A "caller" participant speaks a question (Gemini TTS) into a LiveKit room; the VoiceAgent hears it (energy VAD + Gemini STT), runs a real
research run (live Gemini + web tools), and speaks a short summary back into the room; the caller records that audio and transcribes it.
Writes experiments/voice_demo.json. Needs GOOGLE_API_KEY, LIVEKIT_API_KEY/LIVEKIT_API_SECRET and RESEARCH_OS_TOOLS=live for real sources.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

from research_os.api.app import build_default_service
from research_os.common import load_config, load_dotenv
from research_os.voice.audio import SilenceSegmenter, pcm_to_wav
from research_os.voice.gemini_audio import GeminiSTT, GeminiTTS
from research_os.voice.livekit_agent import STT_RATE, VoiceAgent, make_token

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "What license does LangGraph use?"


async def main() -> dict:
    from livekit import rtc
    load_dotenv()
    cfg = load_config("configs/voice.yaml")
    lk = cfg["livekit"]
    service = build_default_service(load_config("configs/api.yaml"))
    pcm_out: list[bytes] = []
    tts, stt = GeminiTTS(sink=pcm_out.append), GeminiSTT()
    agent = VoiceAgent(service, stt, tts, pcm_out, cfg)
    await agent.connect(lk["url"], make_token("research-agent", lk["room"]))

    caller = rtc.Room()
    heard: asyncio.Future = asyncio.get_running_loop().create_future()

    async def listen(track):
        seg = SilenceSegmenter(STT_RATE, lk["energy_threshold"], lk["silence_ms"], lk["min_speech_ms"])
        async for ev in rtc.AudioStream(track, sample_rate=STT_RATE, num_channels=1):
            utt = seg.push(np.frombuffer(bytes(ev.frame.data), dtype=np.int16))
            if utt is not None and not heard.done():
                heard.set_result(utt)

    caller.on("track_subscribed", lambda t, p, part: asyncio.get_running_loop().create_task(listen(t)) if t.kind == rtc.TrackKind.KIND_AUDIO else None)
    await caller.connect(lk["url"], make_token("caller", lk["room"]))
    source = rtc.AudioSource(tts.sample_rate, 1)
    await caller.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("caller-mic", source),
                                                 rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(1.5)                                                 # let the agent subscribe

    question_pcm = await asyncio.get_running_loop().run_in_executor(None, tts.synthesize, QUESTION)
    quiet = np.zeros(tts.sample_rate * 2, dtype=np.int16)
    samples = np.concatenate([quiet[: tts.sample_rate // 2], np.frombuffer(question_pcm, dtype=np.int16), quiet])
    t0, step = time.time(), tts.sample_rate // 100
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step]
        if len(chunk) < step:
            chunk = np.pad(chunk, (0, step - len(chunk)))
        await source.capture_frame(rtc.AudioFrame(chunk.tobytes(), tts.sample_rate, 1, step))
    reply_pcm = await asyncio.wait_for(heard, timeout=240)
    elapsed = time.time() - t0
    reply_text = await asyncio.get_running_loop().run_in_executor(None, stt.transcribe, pcm_to_wav(reply_pcm, STT_RATE))
    turn = agent.turns[0] if agent.turns else {}
    out = {"question_spoken": QUESTION, "agent_heard_run": turn.get("run_id"), "run_status": turn.get("status"),
           "agent_spoken_text": turn.get("spoken"), "caller_received_transcript": reply_text,
           "caller_received_audio_seconds": round(len(reply_pcm) / 2 / STT_RATE, 1), "seconds_to_spoken_reply": round(elapsed, 1),
           "livekit_url": lk["url"]}
    await caller.disconnect()
    await agent.close()
    service.shutdown()
    return out


if __name__ == "__main__":
    res = asyncio.run(main())
    Path("experiments/voice_demo.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
