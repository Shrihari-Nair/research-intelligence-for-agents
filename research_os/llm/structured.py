"""Schema-validated LLM output with bounded retries (hard rule 5). No free-text parsing."""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from research_os.llm.client import LLMResponse

T = TypeVar("T", bound=BaseModel)
MAX_RETRIES = 2


class MalformedLLMOutput(Exception):
    """The model never produced output matching the schema within the retry bound."""

    def __init__(self, message: str, attempts: int, responses: list[LLMResponse]):
        super().__init__(message)
        self.attempts, self.responses = attempts, responses


def _parse(raw: str, model_cls: type[T]) -> T:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    return model_cls.model_validate(json.loads(cleaned))


def complete_structured(llm, prompt: str, model_cls: type[T], *, model: str | None = None,
                        max_retries: int = MAX_RETRIES, max_tokens: int = 1024, meta: dict | None = None,
                        validate: Callable[[T], None] | None = None) -> tuple[T, list[LLMResponse]]:
    """Ask for JSON matching `model_cls`; on invalid output retry (max `max_retries`) with the error appended.

    Returns (parsed, all responses incl. failed attempts). Raises MalformedLLMOutput when attempts run out.
    """
    responses: list[LLMResponse] = []
    current, last_err = prompt, ""
    for _ in range(max_retries + 1):
        resp = llm.complete(current, model=model, json_mode=True, max_tokens=max_tokens, meta=meta)
        responses.append(resp)
        try:
            parsed = _parse(resp.text, model_cls)
            if validate is not None:
                validate(parsed)  # raise ValueError(message) to request a retry
            return parsed, responses
        except (ValueError, ValidationError) as e:  # json errors are ValueError subclasses
            last_err = str(e)[:500]
            current = (f"{prompt}\n\nYour previous output was invalid ({last_err}). "
                       "Return ONLY valid JSON that matches the required schema, with no commentary.")
    raise MalformedLLMOutput(f"no valid output after {len(responses)} attempts: {last_err}",
                             len(responses), responses)
