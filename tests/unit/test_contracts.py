from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_os.contracts import (
    Claim,
    CostEstimate,
    ExecutionBudget,
    FailureType,
    ResearchPlan,
    ResearchStep,
    RunEvent,
    RunStatus,
    Source,
    canonical_url,
    source_id_for_url,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_canonical_url_normalizes_equivalent_urls():
    a = canonical_url("HTTPS://Example.COM:443/docs/?utm_source=x&b=2&a=1#section")
    b = canonical_url("https://example.com/docs?a=1&b=2")
    assert a == b == "https://example.com/docs?a=1&b=2"
    assert canonical_url("http://example.com") == "http://example.com/"


def test_source_id_is_stable_and_url_equivalence_aware():
    assert source_id_for_url("https://a.com/x?utm_campaign=z") == source_id_for_url("https://a.com/x")
    assert source_id_for_url("https://a.com/x") != source_id_for_url("https://a.com/y")
    assert len(source_id_for_url("https://a.com/x")) == 16


def test_source_requires_valid_type_and_defaults():
    s = Source(source_id="s1", title="t", url="https://a.com", source_type="documentation",
               retrieved_at=NOW, content="hello", tool="fetch_url")
    assert s.published_at is None
    with pytest.raises(ValidationError):
        Source(source_id="s1", title="t", url="https://a.com", source_type="podcast",
               retrieved_at=NOW, content="x", tool="t")


def test_claim_defaults_to_unverified_and_needs_a_quote():
    c = Claim(claim_id="c1", claim="X is fast", source_id="s1", evidence_quote="X is fast",
              extractor_confidence=0.8)
    assert c.verdict == "UNVERIFIED" and c.verdict_rationale is None
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", claim="X", source_id="s1", evidence_quote="", extractor_confidence=0.8)
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", claim="X", source_id="s1", evidence_quote="q", extractor_confidence=1.5)


def test_budget_requires_positive_limits():
    ExecutionBudget(max_steps=5, max_tool_calls=10, max_searches=3, max_llm_calls=8, max_runtime_seconds=60)
    with pytest.raises(ValidationError):
        ExecutionBudget(max_steps=0, max_tool_calls=10, max_searches=3, max_llm_calls=8, max_runtime_seconds=60)


def test_plan_and_step_defaults():
    step = ResearchStep(id="s1", objective="o", agent="search_agent", tools=["web_search"], depends_on=[])
    assert step.status == "pending" and step.attempts == 0 and step.outputs == {}
    plan = ResearchPlan(plan_id="p", workflow="fact_lookup", steps=[step],
                        estimated_cost=CostEstimate(searches=1, sources=3, llm_calls=2))
    assert plan.steps[0].id == "s1"


def test_run_event_and_enums():
    e = RunEvent(run_id="r", seq=0, ts=NOW, node="router", type="started", payload={})
    assert e.type == "started"
    with pytest.raises(ValidationError):
        RunEvent(run_id="r", seq=0, ts=NOW, node="router", type="exploded", payload={})
    assert RunStatus.AWAITING_APPROVAL.value == "AWAITING_APPROVAL"
    assert {f.name for f in FailureType} == {
        "SEARCH_TIMEOUT", "INVALID_SOURCE", "MALFORMED_LLM_OUTPUT", "NO_USEFUL_SOURCES", "CONFLICTING_EVIDENCE",
        "TOOL_FAILURE", "AGENT_FAILURE", "BUDGET_EXHAUSTED", "IMPOSSIBLE_PLAN"}


def test_source_url_must_be_http_or_https():
    from datetime import datetime, timezone

    import pytest
    from pydantic import ValidationError

    from research_os.contracts import Source
    base = dict(source_id="s", title="t", source_type="other", retrieved_at=datetime.now(timezone.utc), content="x", tool="web_search")
    assert Source(url="https://example.com/a", **base).url
    for bad in ("javascript:alert(1)", "data:text/html,x", "file:///etc/passwd", " JavaScript:alert(1)"):
        with pytest.raises(ValidationError):
            Source(url=bad, **base)
