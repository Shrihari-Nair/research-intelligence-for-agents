# Voice interface (M24, optional)

* `research_os/voice/adapters.py`: `STTAdapter` / `TTSAdapter` interfaces, text no-op implementations and
  `voice_turn(stt, service, tts, audio)`: transcript -> normal run -> **short spoken summary** (first sentences of the executive summary,
  markdown and citation markers stripped) plus a pointer to the full report. Failed, cancelled, timed-out and awaiting-approval runs are
  spoken honestly (no internal error text). Voice does **not** change the core pipeline.
* `research_os/voice/gemini_audio.py`: `GeminiSTT` / `GeminiTTS` (same `GOOGLE_API_KEY`; models in `configs/voice.yaml`).
* `research_os/voice/audio.py`: `pcm_to_wav`, energy-based `SilenceSegmenter` (no VAD model download).
* `research_os/voice/livekit_agent.py`: `VoiceAgent` joins a LiveKit room, segments each caller's audio, answers via `voice_turn`, publishes the
  spoken reply as an audio track. Optional extra: `pip install -r requirements-voice.txt`.

## Run the demo

```bash
export LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=$(openssl rand -hex 24)        # throwaway keys for a local dev server; generated, not stored
docker run -d --name researchos-livekit --network host -e LIVEKIT_KEYS="$LIVEKIT_API_KEY: $LIVEKIT_API_SECRET" \
  livekit/livekit-server:latest --dev --bind 127.0.0.1 --node-ip 127.0.0.1      # local only; never expose this server
RESEARCH_OS_TOOLS=live python -m scripts.voice_demo "What license does LangGraph use?"
```

## Verification status

**Verified end to end on 2026-10-03** (`experiments/voice_demo.json`, LiveKit server 1.13.7, SDK `livekit` rtc): a caller participant spoke
"What license does LangGraph use?" (Gemini TTS) into a room; the agent heard it, ran a real research run (live Gemini + web tools, run `f47a283b3aa8`,
SUCCESS) and spoke back an 11.9 s answer ("LangGraph is an open-source library released under the MIT license ..."), which the caller received over
LiveKit and transcribed. Question to start of the full spoken reply: 46.3 s (dominated by the research run).

Not verified: real microphones/speakers or a browser client (the caller is a bot publishing synthesized speech; STT on real human speech, accents
and noise is untested); multiple simultaneous callers; LiveKit Cloud. At interpreter exit the LiveKit SDK prints harmless `FfiHandle.__del__
AssertionError` noise. This is one question, one run, not a latency benchmark.
