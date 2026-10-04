"""Gemini-backed LLMClient (google-genai SDK). Key is read from the environment."""
from __future__ import annotations

import logging
import os
import time

from research_os.llm.client import FALLBACK_INTENT, LLMResponse, build_classify_prompt, parse_llm_intent

log = logging.getLogger(__name__)


class _GenaiBackend:
    """Thin adapter over google-genai so tests can inject a fake."""

    def __init__(self, api_key: str, timeout_s: float):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key,
                                    http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def generate(self, model: str, contents: str, json_mode: bool = False, max_tokens: int = 200):
        t = self._types
        cfg = t.GenerateContentConfig(
            max_output_tokens=max_tokens, temperature=0.0,
            thinking_config=t.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if json_mode else None)
        return self._client.models.generate_content(model=model, contents=contents, config=cfg)


class GeminiLLMClient:
    """Classifies intents / phrases replies with Gemini; never raises on API failure."""

    def __init__(self, model: str, api_key_env: str = "GOOGLE_API_KEY", timeout_s: float = 10.0,
                 backend=None, max_retries: int = 3, sleep=time.sleep):
        self.model = model
        self.max_retries, self._sleep = max_retries, sleep
        if backend is None:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"{api_key_env} is not set; cannot create GeminiLLMClient")
            backend = _GenaiBackend(key, timeout_s)
        self.backend = backend

    _TRANSIENT = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "timed out", "timeout",
                  "disconnected", "ConnectError", "RemoteProtocolError", "ReadError", "Connection reset", "Connection refused")

    def _generate_with_retry(self, **kw):
        """Bounded exponential backoff (1s, 2s, 4s) on transient API errors only; permanent errors raise immediately."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.backend.generate(**kw)
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries or not any(t in str(e) for t in self._TRANSIENT):
                    raise
                self._sleep(float(2 ** attempt))

    def complete(self, prompt: str, *, model: str | None = None, json_mode: bool = False,
                 max_tokens: int = 1024, meta: dict | None = None) -> LLMResponse:
        """Raw completion + usage. Raises on API errors (callers classify them as failures)."""
        name = model or self.model
        t0 = time.perf_counter()
        resp = self._generate_with_retry(model=name, contents=prompt, json_mode=json_mode, max_tokens=max_tokens)
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(text=resp.text or "", model=name,
                           input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                           output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                           latency_ms=(time.perf_counter() - t0) * 1000)

    def classify_intent(self, text: str, intents: list[str]) -> dict:
        try:
            resp = self.backend.generate(model=self.model, contents=build_classify_prompt(text, intents),
                                         json_mode=True, max_tokens=120)
            out = parse_llm_intent(resp.text, intents)
            usage = getattr(resp, "usage_metadata", None)
            out["input_tokens"] = int(getattr(usage, "prompt_token_count", 0) or 0)
            out["output_tokens"] = int(getattr(usage, "candidates_token_count", 0) or 0)
            return out
        except Exception as e:  # network, quota, timeout: degrade, never crash routing
            log.warning("Gemini classify failed: %s", e)
            return {"intent": FALLBACK_INTENT, "reason": f"llm error: {type(e).__name__}",
                    "failed": True, "input_tokens": 0, "output_tokens": 0}

    def generate(self, prompt: str) -> str:
        try:
            return (self.backend.generate(model=self.model, contents=prompt, max_tokens=300).text or "").strip()
        except Exception as e:
            log.warning("Gemini generate failed: %s", e)
            return ""
