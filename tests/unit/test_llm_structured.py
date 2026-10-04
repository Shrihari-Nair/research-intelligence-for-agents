import pytest
from pydantic import BaseModel

from research_os.llm.client import LLMResponse
from research_os.llm.gemini_client import GeminiLLMClient
from research_os.llm.mock_client import ScriptedLLM
from research_os.llm.structured import MalformedLLMOutput, complete_structured


class Answer(BaseModel):
    verdict: str
    score: float


def test_valid_json_parses_first_try():
    llm = ScriptedLLM(['{"verdict": "ok", "score": 0.5}'])
    out, resps = complete_structured(llm, "p", Answer)
    assert out == Answer(verdict="ok", score=0.5) and len(resps) == 1


def test_fenced_json_is_accepted():
    llm = ScriptedLLM(['```json\n{"verdict": "ok", "score": 1}\n```'])
    assert complete_structured(llm, "p", Answer)[0].score == 1.0


def test_invalid_then_valid_retries_with_error_appended():
    llm = ScriptedLLM(['not json', '{"verdict": "ok", "score": 2}'])
    out, resps = complete_structured(llm, "base prompt", Answer)
    assert out.score == 2.0 and len(resps) == 2
    assert "base prompt" in llm.prompts[1] and "invalid" in llm.prompts[1].lower()


def test_schema_violation_counts_as_invalid():
    llm = ScriptedLLM(['{"verdict": "ok"}', '{"verdict": "ok", "score": "high"}', '{"verdict": 1}'])
    with pytest.raises(MalformedLLMOutput) as e:
        complete_structured(llm, "p", Answer)
    assert len(llm.prompts) == 3  # 1 attempt + max 2 retries, never more
    assert e.value.attempts == 3 and len(e.value.responses) == 3


def test_retries_are_bounded_by_argument():
    llm = ScriptedLLM(["bad"] * 10)
    with pytest.raises(MalformedLLMOutput):
        complete_structured(llm, "p", Answer, max_retries=0)
    assert len(llm.prompts) == 1


def test_scripted_llm_exhausted_is_a_loud_error():
    llm = ScriptedLLM([])
    with pytest.raises(RuntimeError, match="exhausted"):
        llm.complete("p")


class _Backend:
    def __init__(self):
        self.kw = None

    def generate(self, **kw):
        self.kw = kw
        r = type("R", (), {})()
        r.text = "hello"
        r.usage_metadata = type("U", (), {"prompt_token_count": 7, "candidates_token_count": 3})()
        return r


def test_gemini_complete_returns_response_with_usage_and_role_model():
    b = _Backend()
    c = GeminiLLMClient(model="default-m", backend=b)
    r = c.complete("hi", model="role-m", json_mode=True, max_tokens=99)
    assert isinstance(r, LLMResponse) and r.text == "hello"
    assert (r.input_tokens, r.output_tokens, r.model) == (7, 3, "role-m")
    assert b.kw["model"] == "role-m" and b.kw["json_mode"] is True and b.kw["max_tokens"] == 99
    assert r.latency_ms >= 0


def test_gemini_retries_transient_errors_with_backoff_then_succeeds():
    class Flaky:
        def __init__(self, fails, exc):
            self.n, self.fails, self.exc = 0, fails, exc

        def generate(self, **kw):
            self.n += 1
            if self.n <= self.fails:
                raise self.exc
            r = type("R", (), {})()
            r.text, r.usage_metadata = "ok", None
            return r
    sleeps = []
    b = Flaky(2, RuntimeError("429 RESOURCE_EXHAUSTED: quota"))
    c = GeminiLLMClient(model="m", backend=b, sleep=sleeps.append)
    assert c.complete("p").text == "ok" and b.n == 3 and sleeps == [1.0, 2.0]


def test_gemini_does_not_retry_permanent_errors_and_gives_up_after_the_bound():
    class Always:
        def __init__(self, exc):
            self.n, self.exc = 0, exc

        def generate(self, **kw):
            self.n += 1
            raise self.exc
    perm = Always(RuntimeError("400 INVALID_ARGUMENT: bad request"))
    c = GeminiLLMClient(model="m", backend=perm, sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        c.complete("p")
    assert perm.n == 1
    transient = Always(RuntimeError("503 UNAVAILABLE"))
    c2 = GeminiLLMClient(model="m", backend=transient, sleep=lambda s: None, max_retries=3)
    with pytest.raises(RuntimeError):
        c2.complete("p")
    assert transient.n == 4                                    # 1 attempt + 3 retries, never more


def test_context_llm_logs_every_call_with_tokens_and_latency():
    from research_os.harness.context import RunContext
    from research_os.llm.mock_client import ScriptedLLM
    from research_os.tools.registry import ToolRegistry
    c = RunContext.create("r", {}, llm=ScriptedLLM(["a b", "c"]), tools=ToolRegistry())
    c.llm.complete("one two three", meta={"prompt": "supervisor.v1"})
    c.llm.complete("x")
    assert [x["prompt"] for x in c.llm.calls] == ["supervisor.v1", None]
    assert c.llm.calls[0]["input_tokens"] == 3 and {"output_tokens", "latency_ms", "model"} <= set(c.llm.calls[0])


@pytest.mark.parametrize("msg", ["Server disconnected without sending a response.", "ConnectError: [Errno 111] Connection refused",
                                 "RemoteProtocolError: peer closed connection", "ReadError", "Connection reset by peer"])
def test_gemini_retries_dropped_connections_too(msg):
    class Flaky:
        n = 0

        def generate(self, **kw):
            Flaky.n += 1
            if Flaky.n == 1:
                raise RuntimeError(msg)
            r = type("R", (), {})()
            r.text, r.usage_metadata = "ok", None
            return r
    Flaky.n = 0
    assert GeminiLLMClient(model="m", backend=Flaky(), sleep=lambda s: None).complete("p").text == "ok" and Flaky.n == 2
