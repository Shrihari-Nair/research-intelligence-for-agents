import json
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from research_os.agents.supervisor import Supervisor
from research_os.api.app import create_app
from research_os.api.service import RunService
from research_os.contracts import CostEstimate, RouterOutput, RoutingMode, Source
from research_os.harness.approvals import ApprovalPolicy, needs_plan_approval
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry

POLICY = ApprovalPolicy(max_searches=3, max_sources=20, max_llm_calls=10, publish=False, expiry_seconds=60)


def test_policy_loads_from_config_and_flags_each_threshold():
    p = ApprovalPolicy.from_config()
    assert p.max_searches > 0 and p.expiry_seconds > 0
    assert needs_plan_approval(CostEstimate(searches=2, sources=10, llm_calls=5), POLICY) == []
    r = needs_plan_approval(CostEstimate(searches=4, sources=25, llm_calls=11), POLICY)
    assert len(r) == 3 and any("searches" in x for x in r) and any("sources" in x for x in r) and any("LLM" in x for x in r)
    assert needs_plan_approval(CostEstimate(searches=3, sources=20, llm_calls=10), POLICY) == []      # at the limit: no approval


class Router:
    def __init__(self, workflow="technology_comparison", intent="TECHNOLOGY_COMPARISON", techs=("A1", "B1", "C1", "D1")):
        self.a = (workflow, intent, list(techs))

    def route(self, text):
        w, i, t = self.a
        return RouterOutput(intent=i, confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow=w, latency_ms=1,
                            entities={"technologies": t, "companies": [], "topics": [], "time_scope": []})


class Agent:
    def __init__(self):
        self.calls = []

    def run(self, step, inputs, ctx, request):
        self.calls.append(step.id)
        if step.agent == "search_agent":
            ctx.tools.call("web_search", {"query": "langgraph"}, ctx, agent="search_agent")
            ctx.sources.add(Source(source_id="s1", title="T", url="https://a.dev/x", source_type="documentation",
                                   retrieved_at=datetime.now(timezone.utc), content="q", tool="web_search"))
        if step.agent == "verifier":
            c = {"claim_id": "c1", "claim": "A claim.", "source_id": "s1", "evidence_quote": "q", "extractor_confidence": 0.9, "verdict": "SUPPORTED",
                 "verdict_rationale": "ok"}
            return {"claims": [c], "conflicts": [], "gate": {"result": "PASS", "reason": None, "failure_type": None}}
        if step.agent == "synthesis_agent":
            return {"summary": "Done.", "findings": [{"statement": "A claim.", "claim_ids": ["c1"]}], "claim_ids": ["c1"]}
        return {"ok": step.id}


def service(tmp_path, policy=POLICY, router=None, agent=None, store=None):
    ag = agent or Agent()
    return RunService(router=router or Router(), supervisor=Supervisor(), agents={n: ag for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")},
                      llm_factory=lambda: ScriptedLLM([]), tools_factory=lambda: build_registry(mode="fixture"), runs_dir=str(tmp_path), llm_backend="mock",
                      max_workers=2, approvals=policy, memory=store), ag


def wait(svc, run_id, statuses, timeout=8):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = svc.get(run_id)
        if r.status.value in statuses:
            return r
        time.sleep(0.02)
    raise AssertionError(f"run stuck in {svc.get(run_id).status}, wanted {statuses}")


def test_expensive_plan_pauses_before_any_step_or_tool_call(tmp_path):
    svc, agent = service(tmp_path)
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    time.sleep(0.3)                                                      # nothing may happen while we wait
    assert agent.calls == [] and rec.ctx.budget.usage()["tool_calls"] == 0 and rec.ctx.budget.usage()["steps"] == 0
    a = rec.detail()["approval"]
    assert a["kind"] == "plan" and a["searches"] > POLICY.max_searches and a["reasons"] and a["budget"]["max_steps"] > 0 and a["expires_at"]
    assert [e.payload["event"] for e in rec.ctx.events.events if e.type == "approval"] == ["requested"]
    assert rec.done is False


def test_approve_resumes_and_completes_the_run(tmp_path):
    svc, agent = service(tmp_path)
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    svc.approve(rec.run_id)
    done = wait(svc, rec.run_id, {"SUCCESS"})
    assert done.status.value == "SUCCESS" and agent.calls and rec.detail()["approval"] is None
    assert [e.payload["event"] for e in rec.ctx.events.events if e.type == "approval"] == ["requested", "approved"]


def test_reject_cancels_cleanly_without_running_anything(tmp_path):
    svc, agent = service(tmp_path)
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    svc.reject(rec.run_id, "too expensive")
    done = wait(svc, rec.run_id, {"CANCELLED"})
    assert agent.calls == [] and done.ctx.budget.usage()["tool_calls"] == 0 and done.done
    rep = done.basic_report()
    assert rep["status"] == "CANCELLED" and "too expensive" in rep["executive_summary"] and rep["findings"] == []
    assert [e.payload["event"] for e in done.ctx.events.events if e.type == "approval"] == ["requested", "rejected"]


def test_cheap_plans_run_straight_through_without_asking(tmp_path):
    svc, agent = service(tmp_path, router=Router("fact_lookup", "FACT_LOOKUP", ()))
    rec = svc.submit("what license does fastapi use")
    wait(svc, rec.run_id, {"SUCCESS"})
    assert agent.calls and not [e for e in rec.ctx.events.events if e.type == "approval"]


def test_pending_approvals_expire_and_cancel_the_run(tmp_path):
    from dataclasses import replace
    svc, agent = service(tmp_path, policy=replace(POLICY, expiry_seconds=0.4))
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    done = wait(svc, rec.run_id, {"CANCELLED"}, timeout=6)
    assert agent.calls == [] and [e.payload["event"] for e in done.ctx.events.events if e.type == "approval"] == ["requested", "expired"]
    assert "expired" in done.basic_report()["executive_summary"]


def test_approving_or_rejecting_a_run_that_is_not_waiting_is_a_conflict(tmp_path):
    svc, _ = service(tmp_path, router=Router("fact_lookup", "FACT_LOOKUP", ()))
    rec = svc.submit("what license does fastapi use")
    wait(svc, rec.run_id, {"SUCCESS"})
    with pytest.raises(ValueError, match="not awaiting"):
        svc.approve(rec.run_id)
    with pytest.raises(ValueError, match="not awaiting"):
        svc.reject(rec.run_id)
    with pytest.raises(KeyError):
        svc.approve("nope")


def test_double_approve_is_a_conflict_not_a_double_run(tmp_path):
    svc, agent = service(tmp_path)
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    svc.approve(rec.run_id)
    with pytest.raises(ValueError):
        svc.approve(rec.run_id)
    wait(svc, rec.run_id, {"SUCCESS"})
    assert agent.calls.count("synthesize") == 1


def test_publish_gate_holds_the_finished_report_until_approved(tmp_path):
    from dataclasses import replace
    svc, agent = service(tmp_path, policy=replace(POLICY, max_searches=99, max_sources=999, max_llm_calls=999, publish=True))
    rec = svc.submit("Compare A1, B1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    a = rec.detail()["approval"]
    assert a["kind"] == "publish" and agent.calls                         # the research ran; only publishing waits
    assert rec.done is False                                              # the report is not released yet
    svc.approve(rec.run_id)
    assert wait(svc, rec.run_id, {"SUCCESS"}).status.value == "SUCCESS"


def test_publish_rejection_cancels_the_run(tmp_path):
    from dataclasses import replace
    svc, _ = service(tmp_path, policy=replace(POLICY, max_searches=99, max_sources=999, max_llm_calls=999, publish=True))
    rec = svc.submit("Compare A1, B1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    svc.reject(rec.run_id, "not ready")
    assert wait(svc, rec.run_id, {"CANCELLED"}).done


def test_http_endpoints_approve_reject_and_status_codes(tmp_path):
    svc, _ = service(tmp_path)
    with TestClient(create_app(service=svc, max_text_chars=200)) as c:
        rid = c.post("/runs", json={"text": "Compare A1, B1, C1 and D1"}).json()["run_id"]
        wait(svc, rid, {"AWAITING_APPROVAL"})
        d = c.get(f"/runs/{rid}").json()
        assert d["status"] == "AWAITING_APPROVAL" and d["approval"]["kind"] == "plan"
        assert c.get(f"/runs/{rid}/report").status_code == 409
        assert c.post(f"/runs/{rid}/approve").status_code == 202
        assert c.post(f"/runs/{rid}/approve").status_code == 409
        wait(svc, rid, {"SUCCESS"})
        rid2 = c.post("/runs", json={"text": "Compare A1, B1, C1 and D1"}).json()["run_id"]
        wait(svc, rid2, {"AWAITING_APPROVAL"})
        assert c.post(f"/runs/{rid2}/reject", json={"reason": "no thanks"}).status_code == 202
        wait(svc, rid2, {"CANCELLED"})
        assert c.get(f"/runs/{rid2}/report").json()["report"]["status"] == "CANCELLED"
        assert c.post("/runs/nope/approve").status_code == 404


def test_sse_streams_approval_events_and_ends_after_resolution(tmp_path):
    svc, _ = service(tmp_path)
    with TestClient(create_app(service=svc, max_text_chars=200)) as c:
        rid = c.post("/runs", json={"text": "Compare A1, B1, C1 and D1"}).json()["run_id"]
        wait(svc, rid, {"AWAITING_APPROVAL"})
        threading.Timer(0.3, lambda: svc.approve(rid)).start()
        body = c.get(f"/runs/{rid}/events").text.replace("\r\n", "\n")
    evs = [json.loads(b.split("data:")[1].strip()) for b in body.strip().split("\n\n") if "data:" in b]
    kinds = [e["payload"].get("event") for e in evs if e["type"] == "approval"]
    assert kinds == ["requested", "approved"] and evs[-1]["node"] == "run" and evs[-1]["payload"]["status"] == "SUCCESS"


def test_persistence_records_the_awaiting_status_and_the_cancellation(tmp_path):
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore
    store = ResearchStore("sqlite://", embedder=HashEmbedder())
    store.create_all()
    svc, _ = service(tmp_path, store=store)
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    assert store.get_run(rec.run_id)["status"] == "AWAITING_APPROVAL"
    svc.reject(rec.run_id)
    wait(svc, rec.run_id, {"CANCELLED"})
    assert store.get_run(rec.run_id)["status"] == "CANCELLED" and store.get_report(rec.run_id)["status"] == "CANCELLED"


def test_awaiting_status_is_persisted_before_it_is_published(tmp_path):
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore
    seen = []

    class Spy(ResearchStore):
        def set_status(self, run_id, status, *a, **k):
            if status == "AWAITING_APPROVAL":
                seen.append(svc.get(run_id).status.value)       # what a poller sees at the moment we persist
            return super().set_status(run_id, status, *a, **k)

    store = Spy("sqlite://", embedder=HashEmbedder())
    store.create_all()
    svc, _ = service(tmp_path, store=store)
    rec = svc.submit("Compare A1, B1, C1 and D1")
    wait(svc, rec.run_id, {"AWAITING_APPROVAL"})
    assert seen == ["RUNNING"]                                  # persisted first; only then visible as AWAITING
    svc.reject(rec.run_id)
