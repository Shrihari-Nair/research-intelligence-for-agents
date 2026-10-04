import ast
import time
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from research_os.contracts import FailureType, Source, source_id_for_url
from research_os.harness.context import RunContext
from research_os.harness.policies import check_fetch_policy
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.registry import ToolError, ToolRegistry, ToolResult, ToolSpec


class In(BaseModel):
    query: str = Field(min_length=1)
    n: int = Field(default=1, ge=1, le=3)


def ctx_for(reg, **cfg):
    c = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=reg)
    return c


def reg_with(fn, **spec_kw):
    reg = ToolRegistry(retry_cfg={"max_retries": 2, "backoff_s": 0.0, "retry_on": ["SEARCH_TIMEOUT", "TOOL_FAILURE"]},
                       permissions={"search_agent": ["research"], "evidence_agent": []})
    reg.register_spec(ToolSpec(name="t", fn=fn, input_schema=In, timeout=spec_kw.pop("timeout", 5), max_retries=spec_kw.pop("max_retries", None),
                               permissions=spec_kw.pop("permissions", ["research"]), cost_class="search", max_content_chars=50, **spec_kw))
    return reg


def source(content="x" * 200, url="https://a.com/p"):
    from datetime import datetime, timezone
    return Source(source_id=source_id_for_url(url), title="t", url=url, source_type="other", retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                  content=content, tool="t")


def test_schema_rejects_bad_arguments_and_coerces_defaults():
    seen = {}
    reg = reg_with(lambda a, c: seen.update(a) or ToolResult())
    c = ctx_for(reg)
    with pytest.raises(ToolError) as e:
        c.tools.call("t", {"query": ""}, c)
    assert e.value.failure_type == FailureType.TOOL_FAILURE and "invalid arguments" in str(e.value)
    with pytest.raises(ToolError):
        c.tools.call("t", {"query": "x", "n": 99}, c)
    c.tools.call("t", {"query": "x"}, c)
    assert seen == {"query": "x", "n": 1}      # defaults filled by the schema


def test_timeout_is_search_timeout_and_the_call_returns_promptly():
    reg = reg_with(lambda a, c: time.sleep(2) or ToolResult(), timeout=0.1, max_retries=0)
    c = ctx_for(reg)
    t0 = time.monotonic()
    with pytest.raises(ToolError) as e:
        c.tools.call("t", {"query": "x"}, c)
    assert e.value.failure_type == FailureType.SEARCH_TIMEOUT and time.monotonic() - t0 < 1.0


def test_retry_then_success_emits_retry_events_and_counts_attempts():
    calls = []

    def flaky(a, c):
        calls.append(1)
        if len(calls) < 3:
            raise ToolError(FailureType.TOOL_FAILURE, "temporary")
        return ToolResult()
    c = ctx_for(reg_with(flaky))
    c.tools.call("t", {"query": "x"}, c)
    assert len(calls) == 3
    assert [e.type for e in c.events.events if e.node == "tool:t"].count("retry") == 2
    assert c.budget.usage()["tool_calls"] == 3


def test_retry_exhaustion_raises_the_last_error():
    calls = []

    def always(a, c):
        calls.append(1)
        raise ToolError(FailureType.SEARCH_TIMEOUT, "slow")
    c = ctx_for(reg_with(always))
    with pytest.raises(ToolError) as e:
        c.tools.call("t", {"query": "x"}, c)
    assert len(calls) == 3 and e.value.failure_type == FailureType.SEARCH_TIMEOUT   # 1 attempt + 2 retries, then stop


def test_non_retryable_failures_are_not_retried():
    calls = []

    def bad_source(a, c):
        calls.append(1)
        raise ToolError(FailureType.INVALID_SOURCE, "404")
    c = ctx_for(reg_with(bad_source))
    with pytest.raises(ToolError):
        c.tools.call("t", {"query": "x"}, c)
    assert len(calls) == 1


def test_permission_denied_for_agents_without_the_tool_permission():
    c = ctx_for(reg_with(lambda a, ctx: ToolResult()))
    c.tools.call("t", {"query": "x"}, c, agent="search_agent")
    with pytest.raises(ToolError) as e:
        c.tools.call("t", {"query": "x"}, c, agent="evidence_agent")
    assert "permission" in str(e.value).lower()
    assert any(ev.type == "failed" and "permission" in str(ev.payload) for ev in c.events.events)
    with pytest.raises(ToolError):
        c.tools.call("t", {"query": "x"}, c, agent="unknown_agent")


def test_output_is_capped_per_source():
    c = ctx_for(reg_with(lambda a, ctx: ToolResult(sources=[source("y" * 500)])))
    out = c.tools.call("t", {"query": "x"}, c)
    assert len(out.sources[0].content) == 50


def test_policy_blocks_and_allows_domains():
    pol = {"allowed_domains": ["docs.python.org"], "blocked_domains": ["evil.com"]}
    check_fetch_policy("https://docs.python.org/3/", pol)
    check_fetch_policy("https://sub.docs.python.org/x", pol)
    with pytest.raises(ToolError):
        check_fetch_policy("https://example.com/", pol)
    with pytest.raises(ToolError):
        check_fetch_policy("https://evil.com/x", {"allowed_domains": [], "blocked_domains": ["evil.com"]})
    with pytest.raises(ToolError):
        check_fetch_policy("https://a.evil.com/x", {"allowed_domains": [], "blocked_domains": ["evil.com"]})
    check_fetch_policy("https://example.com/", {"allowed_domains": [], "blocked_domains": []})


def test_fetch_url_applies_domain_policy_through_the_registry():
    cfg = {"mode": "fixture", "fixtures_dir": "tests/fixtures/corpus", "limits": {"max_results": 10, "max_content_chars": 20000},
           "policy": {"allowed_domains": ["only-this.example"], "blocked_domains": []},
           "registry": {"default_timeout_s": 5, "max_retries": 0, "backoff_s": 0.0, "retry_on": []}, "permissions": {"source_agent": ["research"]}}
    reg = build_registry(cfg)
    c = ctx_for(reg)
    with pytest.raises(ToolError) as e:
        c.tools.call("fetch_url", {"url": "https://docs.example-db.dev/raft"}, c, agent="source_agent")
    assert e.value.failure_type == FailureType.INVALID_SOURCE and "policy" in str(e.value).lower()


def test_builtin_tools_have_schemas_and_agents_are_permitted_per_config():
    reg = build_registry(mode="fixture")
    c = ctx_for(reg)
    with pytest.raises(ToolError):
        c.tools.call("web_search", {}, c, agent="search_agent")                 # query is required
    assert c.tools.call("web_search", {"query": "langgraph"}, c, agent="search_agent").sources
    with pytest.raises(ToolError):
        c.tools.call("web_search", {"query": "langgraph"}, c, agent="verifier")  # verifier has no research permission


def test_agent_modules_never_import_tool_implementations_directly():
    allowed = {"research_os.tools", "research_os.tools.registry"}
    offenders = []
    for path in Path("research_os/agents").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            for m in mods:
                if (m.startswith("research_os.tools") and m not in allowed) or m.split(".")[0] in ("httpx", "requests", "urllib3"):
                    offenders.append(f"{path}: {m}")
    assert offenders == [], offenders
