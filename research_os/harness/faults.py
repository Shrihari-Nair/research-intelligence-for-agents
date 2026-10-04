"""Deterministic fault injection for the tool registry and the LLM (used by evaluation/failure_injection.py)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from research_os.contracts import Source, source_id_for_url
from research_os.llm.client import LLMResponse
from research_os.tools.registry import ToolError, ToolResult


def garbage_source(url: str) -> Source:
    """A syntactically valid Source whose text is unusable (binary-looking noise, no sentences)."""
    now = datetime.now(timezone.utc)
    return Source(source_id=source_id_for_url(url), title="???", url=url, source_type="other", published_at=now, retrieved_at=now,
                  content="ÿØÿà �� \x01\x02 ### @@@ %%% 0x00 0x1f ~~~ ^^^ ", tool="web_search")


class FaultyRegistry:
    """Proxy around a ToolRegistry that injects failures per tool. Everything else is delegated untouched."""

    def __init__(self, inner):
        self._inner = inner
        self._rules: dict[str, list[dict]] = {}
        self._calls: dict[str, int] = {}
        self.injected: dict[str, int] = {}

    def names(self) -> list[str]:
        return self._inner.names()

    def _rule(self, tool: str, **rule) -> None:
        self._rules.setdefault(tool, []).append({"seen": 0, **rule})

    def fail(self, tool: str, error: ToolError, first_n: int | None = None) -> None:
        """Raise `error` on the first `first_n` calls (all calls if None)."""
        self._rule(tool, kind="fail", error=error, first_n=first_n)

    def empty(self, tool: str) -> None:
        """Return no sources on every call."""
        self._rule(tool, kind="result", result=ToolResult(), first_n=None)

    def replace(self, tool: str, result: ToolResult, first_n: int | None = None) -> None:
        """Return `result` instead of the real answer."""
        self._rule(tool, kind="result", result=result, first_n=first_n)

    def tamper_args(self, tool: str, args: dict, first_n: int = 1) -> None:
        """Replace the call arguments (e.g. with invalid ones) on the first `first_n` calls."""
        self._rule(tool, kind="args", args=args, first_n=first_n)

    def call(self, name: str, args: dict, ctx, agent: str | None = None) -> ToolResult:
        for rule in self._rules.get(name, []):
            if rule["first_n"] is not None and rule["seen"] >= rule["first_n"]:
                continue
            rule["seen"] += 1
            self.injected[name] = self.injected.get(name, 0) + 1
            if rule["kind"] == "fail":
                ctx.events.emit(f"tool:{name}", "failed", {"failure_type": rule["error"].failure_type.value, "error": f"injected: {rule['error']}"})
                raise rule["error"]
            if rule["kind"] == "result":
                ctx.events.emit(f"tool:{name}", "info", {"injected": "replaced result"})
                for s in rule["result"].sources:
                    ctx.sources.add(s)
                return rule["result"]
            if rule["kind"] == "args":
                args = rule["args"]
        return self._inner.call(name, args, ctx, agent=agent)


def _prompt_id(meta: dict | None) -> str:
    return (meta or {}).get("prompt") or ""


class FaultyLLM:
    """Proxy around an LLM client that corrupts or rewrites responses for chosen prompts."""

    def __init__(self, inner):
        self._inner = inner
        self._malformed: list[dict] = []
        self._extra_claim: str | None = None
        self.injected = 0

    def malformed(self, prompt_prefix: str, first_n: int) -> None:
        """Return non-JSON text for the first `first_n` calls whose prompt id starts with `prompt_prefix`."""
        self._malformed.append({"prefix": prompt_prefix, "left": first_n})

    def add_overreaching_claim(self, claim: str) -> None:
        """Put an overreaching claim (quoted verbatim from the first passage) FIRST in every evidence-extraction answer (so claim caps cannot drop it)."""
        self._extra_claim = claim

    def complete(self, prompt: str, *, model=None, json_mode=False, max_tokens=1024, meta=None) -> LLMResponse:
        pid = _prompt_id(meta)
        for m in self._malformed:
            if pid.startswith(m["prefix"]) and m["left"] > 0:
                m["left"] -= 1
                self.injected += 1
                return LLMResponse(text='{"claims": [ {"claim": "truncated', model=model or "faulty")
        resp = self._inner.complete(prompt, model=model, json_mode=json_mode, max_tokens=max_tokens, meta=meta)
        if self._extra_claim and pid.startswith("evidence"):
            m = re.search(r"<<<SOURCE id=(\S+)\n(.*?)\nSOURCE>>>", prompt, re.S)
            if m:
                data = json.loads(resp.text)
                first = re.split(r"(?<=[.!?])\s", m.group(2).strip())[0]
                quote = first[:50].rsplit(" ", 1)[0] if len(first) > 50 else first   # a fragment: its (source, quote) key differs from the real claim
                data.setdefault("claims", []).insert(0, {"claim": self._extra_claim, "source_id": m.group(1), "evidence_quote": quote, "confidence": 0.9})
                self.injected += 1
                return LLMResponse(text=json.dumps(data), input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, model=resp.model)
        return resp
