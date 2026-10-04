import json
import time

import pytest
from fastapi.testclient import TestClient

from research_os.agents.supervisor import Supervisor
from research_os.api.app import create_app
from research_os.api.service import RunService
from research_os.contracts import RouterOutput, RoutingMode, Source
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry


class FakeRouter:
    def route(self, text):
        return RouterOutput(intent="GENERAL_RESEARCH", confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow="general_research",
                            latency_ms=1.0, entities={"technologies": [], "companies": [], "topics": [], "time_scope": []})


class OkAgent:
    def run(self, step, inputs, ctx, request):
        if step.agent == "search_agent":
            from datetime import datetime, timezone
            ctx.sources.add(Source(source_id="s1", title="Doc", url="https://docs.x.dev/a", source_type="documentation",
                                   retrieved_at=datetime.now(timezone.utc), content="q", tool="web_search"))
        if step.agent == "verifier":
            claim = {"claim_id": "c1", "claim": "A supported claim.", "source_id": "s1", "evidence_quote": "q",
                     "extractor_confidence": 0.9, "verdict": "SUPPORTED", "verdict_rationale": "ok"}
            return {"claims": [claim], "conflicts": [], "gate": {"result": "PASS", "reason": None, "failure_type": None}}
        if step.agent == "synthesis_agent":
            return {"summary": "All good.", "findings": [{"statement": "A supported claim.", "claim_ids": ["c1"]}], "claim_ids": ["c1"]}
        return {"ok": step.id}


class BoomRouter:
    def route(self, text):
        raise RuntimeError("router exploded")


def make_service(tmp_path, router=None, agents=None):
    ag = agents or {n: OkAgent() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    return RunService(router=router or FakeRouter(), supervisor=Supervisor(), agents=ag, llm_factory=lambda: ScriptedLLM([]),
                      tools_factory=lambda: build_registry(mode="fixture"), runs_dir=str(tmp_path), llm_backend="mock", max_workers=2)


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(service=make_service(tmp_path), cors_origins=["http://localhost:3000"], max_text_chars=200)) as c:
        yield c


def wait_done(client, run_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f"/runs/{run_id}").json()
        if d["status"] in ("SUCCESS", "PARTIAL", "FAILED", "CANCELLED"):
            return d
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_create_run_returns_id_and_the_run_completes_with_a_full_detail_view(client):
    r = client.post("/runs", json={"text": "tell me about raft"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    d = wait_done(client, run_id)
    assert d["status"] == "SUCCESS" and d["workflow"] == "general_research" and d["llm_backend"] == "mock"
    assert [s["id"] for s in d["plan"]] == ["search", "read", "extract", "verify", "synthesize"]
    assert all(s["status"] == "succeeded" for s in d["plan"])
    assert d["router"]["intent"] == "GENERAL_RESEARCH" and d["claims"][0]["claim_id"] == "c1"
    assert d["budget"]["steps"] == 5 and d["budget"]["limits"]["max_steps"] > 0 and d["error"] is None


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": "x" * 201}, {"text": 5}, {"text": "ok", "budget_profile": "huge"}])
def test_validation_errors_are_422(client, payload):
    assert client.post("/runs", json=payload).status_code == 422


def test_unknown_run_is_404_everywhere(client):
    for path in ("/runs/nope", "/runs/nope/report", "/runs/nope/events"):
        assert client.get(path).status_code == 404


def test_history_lists_runs_newest_first_with_summary_fields(client):
    ids = []
    for t in ("first question", "second question"):
        ids.append(client.post("/runs", json={"text": t}).json()["run_id"])
        wait_done(client, ids[-1])
        time.sleep(0.01)
    rows = client.get("/runs").json()
    assert [r["run_id"] for r in rows][:2] == [ids[1], ids[0]]
    assert {"run_id", "request", "status", "workflow", "created_at", "duration_s", "n_sources", "n_supported_claims", "n_conflicts"} <= set(rows[0])


def parse_sse(text):
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        ev = {}
        for line in block.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                ev[k.strip()] = v.strip()
        if "data" in ev:
            events.append((int(ev["id"]), json.loads(ev["data"])))
    return events


def test_sse_streams_all_events_and_ends_when_the_run_finishes(client):
    run_id = client.post("/runs", json={"text": "tell me about raft"}).json()["run_id"]
    with client.stream("GET", f"/runs/{run_id}/events") as r:
        assert r.status_code == 200 and "text/event-stream" in r.headers["content-type"]
        body = "".join(r.iter_text())
    events = parse_sse(body)
    seqs = [s for s, _ in events]
    assert seqs == list(range(len(seqs))) and len(seqs) > 8
    nodes = [e["node"] for _, e in events]
    assert "router" in nodes and "planner" in nodes and "step:verify" in nodes
    assert events[-1][1]["node"] == "run" and events[-1][1]["payload"]["status"] == "SUCCESS"


def test_sse_resumes_from_seq_query_and_last_event_id_header(client):
    run_id = client.post("/runs", json={"text": "tell me about raft"}).json()["run_id"]
    wait_done(client, run_id)
    full = parse_sse(client.get(f"/runs/{run_id}/events").text)
    tail = parse_sse(client.get(f"/runs/{run_id}/events?from_seq=5").text)
    assert len(full) > 10 and [s for s, _ in tail] == [s for s, _ in full if s >= 5] and tail[0][0] == 5
    resumed = parse_sse(client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "5"}).text)
    assert [s for s, _ in resumed] == [s for s, _ in full if s >= 6] and resumed[0][0] == 6


def test_report_json_for_success_and_409_while_running(tmp_path):
    import threading
    gate = threading.Event()

    class Slow(OkAgent):
        def run(self, step, inputs, ctx, request):
            if step.agent == "search_agent":
                gate.wait(5)
            return super().run(step, inputs, ctx, request)
    ag = {n: (Slow() if n == "search_agent" else OkAgent()) for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    with TestClient(create_app(service=make_service(tmp_path, agents=ag), max_text_chars=200)) as c:
        run_id = c.post("/runs", json={"text": "slow run"}).json()["run_id"]
        assert c.get(f"/runs/{run_id}/report").status_code == 409
        gate.set()
        wait_done(c, run_id)
        rep = c.get(f"/runs/{run_id}/report").json()
        assert rep["status"] == "SUCCESS" and rep["report"]["executive_summary"] == "All good." and rep["report"]["findings"][0]["claim_ids"] == ["c1"]


def test_a_crashing_run_is_marked_failed_and_the_server_keeps_working(tmp_path):
    with TestClient(create_app(service=make_service(tmp_path, router=BoomRouter()), max_text_chars=200)) as c:
        run_id = c.post("/runs", json={"text": "boom"}).json()["run_id"]
        d = wait_done(c, run_id)
        assert d["status"] == "FAILED" and "router exploded" in d["error"]
        assert c.get("/health").status_code == 200
        events = parse_sse(c.get(f"/runs/{run_id}/events").text)
        assert events[-1][1]["type"] == "failed" and events[-1][1]["node"] == "run"


def test_cors_allows_the_configured_origin(client):
    r = client.options("/runs", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ---------------- persistence across restarts (M16) ----------------
def test_history_detail_and_report_survive_a_server_restart(tmp_path):
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore

    store = ResearchStore("sqlite://", embedder=HashEmbedder())
    store.create_all()

    def service():
        return RunService(router=FakeRouter(), supervisor=Supervisor(),
                          agents={n: OkAgent() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")},
                          llm_factory=lambda: ScriptedLLM([]), tools_factory=lambda: build_registry(mode="fixture", memory=store),
                          runs_dir=str(tmp_path), llm_backend="mock", max_workers=2, memory=store)
    with TestClient(create_app(service=service(), max_text_chars=200)) as c1:
        run_id = c1.post("/runs", json={"text": "persist me"}).json()["run_id"]
        wait_done(c1, run_id)
    assert store.get_run(run_id)["status"] == "SUCCESS"
    with TestClient(create_app(service=service(), max_text_chars=200)) as c2:          # a "restarted" server: empty in-memory registry
        rows = c2.get("/runs").json()
        assert [r["run_id"] for r in rows] == [run_id] and rows[0]["status"] == "SUCCESS"
        d = c2.get(f"/runs/{run_id}").json()
        assert d["request"] == "persist me" and d["status"] == "SUCCESS" and d["workflow"] == "general_research"
        rep = c2.get(f"/runs/{run_id}/report").json()
        assert rep["status"] == "SUCCESS" and rep["report"]["executive_summary"] == "All good."
        assert c2.get("/runs/doesnotexist").status_code == 404


def test_a_crashing_run_is_persisted_as_failed(tmp_path):
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore

    store = ResearchStore("sqlite://", embedder=HashEmbedder())
    store.create_all()
    svc = make_service(tmp_path, router=BoomRouter())
    svc.memory = store
    with TestClient(create_app(service=svc, max_text_chars=200)) as c:
        run_id = c.post("/runs", json={"text": "boom"}).json()["run_id"]
        wait_done(c, run_id)
    assert store.get_run(run_id)["status"] == "FAILED" and "router exploded" in store.get_report(run_id)["executive_summary"]


def test_report_formats_json_markdown_and_pdf(client):
    run_id = client.post("/runs", json={"text": "tell me about raft"}).json()["run_id"]
    wait_done(client, run_id)
    js = client.get(f"/runs/{run_id}/report?format=json").json()["report"]
    assert js["question"] == "tell me about raft" and js["findings"][0]["claim_ids"] == ["c1"] and js["evidence"][0]["source_id"] == "s1"
    md = client.get(f"/runs/{run_id}/report?format=md")
    assert md.status_code == 200 and "text/markdown" in md.headers["content-type"] and "## Findings" in md.text and "[1]" in md.text
    pdf = client.get(f"/runs/{run_id}/report?format=pdf")
    assert pdf.status_code in (200, 501)
    if pdf.status_code == 200:
        assert pdf.content[:5] == b"%PDF-" and pdf.headers["content-type"] == "application/pdf"
    assert client.get(f"/runs/{run_id}/report?format=docx").status_code == 422


def test_runs_are_traced_when_a_tracer_is_configured_and_trace_url_is_exposed(tmp_path):
    from research_os.observability.tracing import LangfuseTracer
    from tests.unit.test_tracing import FakeLangfuse

    lf = FakeLangfuse()
    svc = make_service(tmp_path)
    svc.tracer_factory = lambda: LangfuseTracer(lf)
    with TestClient(create_app(service=svc, max_text_chars=200)) as c:
        run_id = c.post("/runs", json={"text": "trace me"}).json()["run_id"]
        d = wait_done(c, run_id)
    assert d["status"] == "SUCCESS" and lf.roots and lf.roots[0].name == "research_run" and lf.roots[0].ended
    assert dict(lf.scores)["run_success"] == 1.0 and lf.flushed >= 1
    assert d["trace_url"] == "http://langfuse.example/trace/trace123"


def test_source_content_endpoint_returns_full_text_for_live_and_persisted_runs(tmp_path):
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore

    store = ResearchStore("sqlite://", embedder=HashEmbedder())
    store.create_all()
    svc = make_service(tmp_path)
    svc.memory = store
    with TestClient(create_app(service=svc, max_text_chars=200)) as c:
        run_id = c.post("/runs", json={"text": "get my sources"}).json()["run_id"]
        wait_done(c, run_id)
        r = c.get(f"/runs/{run_id}/sources/s1")
        assert r.status_code == 200 and r.json()["source_id"] == "s1" and r.json()["content"] == "q" and r.json()["url"] == "https://docs.x.dev/a"
        assert c.get(f"/runs/{run_id}/sources/nope").status_code == 404
        assert c.get("/runs/nope/sources/s1").status_code == 404
    svc2 = make_service(tmp_path)
    svc2.memory = store
    with TestClient(create_app(service=svc2, max_text_chars=200)) as c2:                 # "restarted" server
        assert c2.get(f"/runs/{run_id}/sources/s1").json()["content"] == "q"


def test_cors_origins_can_be_overridden_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_OS_CORS_ORIGINS", "http://localhost:3101, http://example.test")
    with TestClient(create_app(service=make_service(tmp_path), max_text_chars=200)) as c:
        for origin in ("http://localhost:3101", "http://example.test"):
            r = c.options("/runs", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
            assert r.headers.get("access-control-allow-origin") == origin
        r = c.options("/runs", headers={"Origin": "http://evil.test", "Access-Control-Request-Method": "POST"})
        assert r.headers.get("access-control-allow-origin") is None


def test_a_run_is_never_reported_done_before_it_is_persisted(tmp_path):
    """Regression: the terminal status used to be published before persistence, so pollers could read an empty store."""
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore

    class SlowStore(ResearchStore):
        def finish_run(self, *a, **kw):
            import time as t
            t.sleep(0.3)                                   # widen the window the old ordering lost
            return super().finish_run(*a, **kw)
    store = SlowStore("sqlite://", embedder=HashEmbedder())
    store.create_all()
    svc = make_service(tmp_path, router=BoomRouter())
    svc.memory = store
    with TestClient(create_app(service=svc, max_text_chars=200)) as c:
        run_id = c.post("/runs", json={"text": "boom"}).json()["run_id"]
        wait_done(c, run_id)
        assert store.get_run(run_id)["status"] == "FAILED"           # visible the instant the API says FAILED
