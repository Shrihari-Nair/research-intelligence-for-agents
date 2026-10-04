import os
from datetime import datetime, timedelta, timezone

import pytest

from research_os.contracts import Source, source_id_for_url
from research_os.memory.embeddings import HashEmbedder
from research_os.memory.research_store import ResearchStore

URLS = [("sqlite", "sqlite://")]
if os.environ.get("TEST_POSTGRES_URL"):
    URLS.append(("postgres", os.environ["TEST_POSTGRES_URL"]))


@pytest.fixture(params=[u for _, u in URLS], ids=[n for n, _ in URLS])
def store(request):
    s = ResearchStore(request.param, embedder=HashEmbedder())
    s.create_all()
    yield s
    s.drop_all()


NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def src(url="https://docs.x.dev/a", content="LangGraph supports persistence through checkpointers. It is MIT licensed."):
    return Source(source_id=source_id_for_url(url), title="Doc", url=url, source_type="documentation", published_at=NOW,
                  retrieved_at=NOW, content=content, tool="web_search")


def claim(cid="c1", text="LangGraph supports persistence", quote="LangGraph supports persistence through checkpointers",
          source=None, verdict="SUPPORTED"):
    s = source or src()
    return {"claim_id": cid, "claim": text, "source_id": s.source_id, "evidence_quote": quote, "extractor_confidence": 0.9,
            "verdict": verdict, "verdict_rationale": "ok"}


def finished_run(store, run_id="r1", text="What does LangGraph persistence do?", claims=None, created=NOW, status="SUCCESS", report=None):
    store.create_run(run_id, text, llm_backend="mock", created_at=created)
    s = src()
    store.add_sources(run_id, [s])
    store.add_claims(run_id, claims if claims is not None else [claim()])
    store.finish_run(run_id, status, workflow="general_research", budget={"steps": 5}, router={"intent": "GENERAL_RESEARCH"},
                     finished_at=created, report=report or {"summary": "LangGraph can persist state.", "findings": []})
    return s


def test_run_round_trip_with_sources_claims_and_report(store):
    finished_run(store)
    run = store.get_run("r1")
    assert run["request"].startswith("What does LangGraph") and run["status"] == "SUCCESS" and run["workflow"] == "general_research"
    assert run["budget"] == {"steps": 5} and run["router"]["intent"] == "GENERAL_RESEARCH"
    assert [c["claim_id"] for c in store.get_claims("r1")] == ["c1"] and store.get_claims("r1")[0]["verdict"] == "SUPPORTED"
    assert store.get_sources("r1")[0].url == "https://docs.x.dev/a"
    assert store.get_report("r1")["summary"] == "LangGraph can persist state."
    assert store.get_run("nope") is None


def test_partial_runs_are_persisted_too(store):
    finished_run(store, "p1", status="PARTIAL", report={"status": "PARTIAL", "message": "stopped early"})
    assert store.get_run("p1")["status"] == "PARTIAL" and store.get_report("p1")["message"] == "stopped early"


def test_sources_are_deduplicated_across_runs_and_linked_to_each(store):
    finished_run(store, "r1")
    store.create_run("r2", "another question", llm_backend="mock", created_at=NOW)
    store.add_sources("r2", [src()])
    assert store.count_sources() == 1
    assert len(store.get_sources("r1")) == 1 and len(store.get_sources("r2")) == 1


def test_incremental_persistence_keeps_data_when_a_run_never_finishes(store):
    store.create_run("crash", "a doomed run", llm_backend="mock", created_at=NOW)
    store.add_sources("crash", [src()])
    store.add_claims("crash", [claim("c9")])
    assert store.get_run("crash")["status"] == "RUNNING"          # never finished
    assert [c["claim_id"] for c in store.get_claims("crash")] == ["c9"] and len(store.get_sources("crash")) == 1


def test_claims_are_upserted_not_duplicated(store):
    finished_run(store)
    store.add_claims("r1", [claim("c1", verdict="CONTRADICTED")])
    assert len(store.get_claims("r1")) == 1 and store.get_claims("r1")[0]["verdict"] == "CONTRADICTED"


def test_list_runs_newest_first(store):
    finished_run(store, "old", created=NOW - timedelta(days=2))
    finished_run(store, "new", created=NOW)
    assert [r["run_id"] for r in store.list_runs()] == ["new", "old"]


def test_recall_finds_the_earlier_question_and_its_report(store):
    finished_run(store, "r1", text="What does LangGraph persistence do?")
    finished_run(store, "r2", text="How big is the vector database market?", claims=[])
    hits = store.search("what did we find about langgraph persistence", kinds=("questions",), k=3, min_similarity=0.3)
    assert hits[0]["run_id"] == "r1" and hits[0]["kind"] == "questions" and hits[0]["similarity"] > 0.3
    assert hits[0]["report"]["summary"] == "LangGraph can persist state."
    assert all(h["run_id"] != "r2" for h in hits)


def test_search_over_claims_and_reports_and_thresholds(store):
    finished_run(store)
    c = store.search("langgraph persistence checkpointers", kinds=("claims",), k=3, min_similarity=0.2)
    assert c and c[0]["kind"] == "claims" and "persistence" in c[0]["text"]
    assert store.search("completely unrelated astrophysics", kinds=("claims", "questions", "reports"), k=3, min_similarity=0.5) == []
    assert store.search("langgraph", kinds=("reports",), k=3, min_similarity=0.1)[0]["kind"] == "reports"


def test_prior_findings_respect_age_verdict_and_similarity(store):
    finished_run(store, "fresh", created=NOW)
    finished_run(store, "stale", created=NOW - timedelta(days=90), claims=[claim("old1", "LangGraph supports persistence")])
    got = store.prior_findings("does langgraph support persistence", now=NOW + timedelta(days=1), max_age_days=30,
                               min_similarity=0.3, k=8)
    ids = [g["claim"]["claim_id"] for g in got]
    assert "c1" in ids and "old1" not in ids                       # stale claims are not reused


def test_prior_findings_exclude_unsupported_claims(store):
    finished_run(store, "r1", claims=[claim("bad", "LangGraph is faster", verdict="UNSUPPORTED"), claim("good")])
    got = store.prior_findings("langgraph", now=NOW, max_age_days=30, min_similarity=0.1, k=8)
    assert [g["claim"]["claim_id"] for g in got] == ["good"]


def test_reused_claims_must_still_have_their_source_and_verbatim_quote(store):
    s = src()
    finished_run(store, "r1", claims=[claim("ok"), claim("fab", quote="LangGraph is 100 times faster", text="LangGraph is faster")])
    got = store.prior_findings("langgraph", now=NOW, max_age_days=30, min_similarity=0.1, k=8)
    assert [g["claim"]["claim_id"] for g in got] == ["ok"]
    assert got[0]["source"]["url"] == s.url and got[0]["run_id"] == "r1" and got[0]["age_days"] == 0


def test_monitoring_diff_lists_new_and_unchanged_findings(store):
    finished_run(store, "prev", text="monitor langgraph releases", claims=[claim("p1", "LangGraph supports persistence")])
    new = [claim("n1", "LangGraph supports persistence"), claim("n2", "LangGraph added a new streaming mode")]
    d = store.diff_with_prior("monitor langgraph releases", new, now=NOW, similarity=0.85, exclude_run_id="current")
    assert d["prior_run_id"] == "prev"
    assert [c["claim_id"] for c in d["unchanged"]] == ["n1"] and [c["claim_id"] for c in d["new"]] == ["n2"]


def test_monitoring_diff_with_no_prior_run_marks_everything_new(store):
    d = store.diff_with_prior("monitor something", [claim("n1")], now=NOW, similarity=0.85, exclude_run_id="x")
    assert d["prior_run_id"] is None and [c["claim_id"] for c in d["new"]] == ["n1"] and d["unchanged"] == []
