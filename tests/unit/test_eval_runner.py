import json

import pytest

from evaluation.runner import DETERMINISTIC_KEYS, load_run, run_one, run_tasks
from evaluation.validate_dataset import Task
from research_os.contracts import RouterOutput, RoutingMode
from research_os.llm.demo_client import DemoLLM
from research_os.tools.fixtures import FixtureCorpus


class FakeRouter:
    def __init__(self, conf=0.9):
        self.conf = conf

        class E:
            @staticmethod
            def extract(text):
                return {"technologies": ["LangGraph", "CrewAI"], "companies": [], "topics": [], "time_scope": []}
        self.entities = E()

    def route(self, text):
        mode = RoutingMode.DIRECT if self.conf >= 0.6 else RoutingMode.LLM_FALLBACK
        return RouterOutput(intent="TECHNOLOGY_COMPARISON", confidence=self.conf, routing_mode=mode, workflow="technology_comparison",
                            latency_ms=1.0, entities=self.entities.extract(text))


def task(id="t1", text="Compare LangGraph and CrewAI"):
    return Task(id=id, input=text, expected_intent="TECHNOLOGY_COMPARISON", expected_workflow="technology_comparison",
                required_source_types=["documentation"], min_supported_claims=2, kind="normal")


@pytest.fixture()
def cassettes(tmp_path):
    corpus = FixtureCorpus.load("tests/fixtures/corpus")
    d = tmp_path / "cas"
    d.mkdir()
    for tid in ("t1", "t2"):
        (d / f"{tid}.sources.json").write_text(json.dumps([s.model_dump(mode="json") for s in
                                                          (corpus._source(doc) for doc in corpus.docs)]))
        (d / f"{tid}.jsonl").write_text("")
    return d


def cfg(tmp_path, cas):
    return {"paths": {"cassettes": str(cas), "llm_cache": str(tmp_path / "llm.jsonl"), "runs": str(tmp_path / "runs"),
                      "results": str(tmp_path)}, "seed": 42,
            "pricing_per_mtok": {"input": 0.3, "output": 2.5}}


def test_run_record_has_everything_the_metrics_need(tmp_path, cassettes):
    rec = run_one("D", task(), cfg(tmp_path, cassettes), FakeRouter(), base_llm=DemoLLM())
    assert rec["system"] == "D" and rec["task_id"] == "t1" and rec["status"] in ("SUCCESS", "PARTIAL")
    assert rec["workflow"] == "technology_comparison" and rec["router_intent"] == "TECHNOLOGY_COMPARISON"
    assert rec["llm"]["calls"] >= 3 and rec["llm"]["input_tokens"] > 0 and rec["tools"]["calls"] > 0 and rec["tools"]["searches"] > 0
    assert rec["report"]["question"] == "Compare LangGraph and CrewAI" and rec["report"]["evidence"]
    assert all({"claim_id", "claim", "quote", "source_title", "source_type", "verbatim"} <= set(e) for e in rec["evidence_checks"])
    assert all(e["verbatim"] for e in rec["evidence_checks"]) and rec["wall_ms"] > 0 and set(rec["tools"]["by_name"]) <= {
        "web_search", "github_search", "arxiv_search", "pubmed_search", "fetch_url"}


def test_system_A_records_the_supervisor_llm_latency_and_B_does_not(tmp_path, cassettes):
    import json as j

    class SupervisorAware(DemoLLM):
        def _respond(self, prompt):
            if "You are the supervisor" in prompt:
                return {"workflow": "technology_comparison", "rationale": "x"}
            return super()._respond(prompt)
    a = run_one("A", task(), cfg(tmp_path, cassettes), FakeRouter(), base_llm=SupervisorAware())
    b = run_one("B", task("t2"), cfg(tmp_path, cassettes), FakeRouter(0.2), base_llm=SupervisorAware())
    assert a["supervisor_llm_used"] is True and a["routing"]["llm_calls"] == 1 and j.dumps(a["routing"])
    assert b["supervisor_llm_used"] is False and b["routing"]["llm_calls"] == 0


def test_second_offline_run_from_the_cache_is_identical_on_every_deterministic_field(tmp_path, cassettes):
    c = cfg(tmp_path, cassettes)
    first = run_one("D", task(), c, FakeRouter(), base_llm=DemoLLM())
    second = run_one("D", task(), c, FakeRouter(), base_llm=None)         # offline: any cache miss would raise
    for k in DETERMINISTIC_KEYS:
        assert first[k] == second[k], k
    assert second["llm"]["cache"] == {"hits": first["llm"]["calls"], "misses": 0}


def test_offline_without_a_cache_fails_loudly_not_silently(tmp_path, cassettes):
    rec = run_one("C", task(), cfg(tmp_path, cassettes), FakeRouter(), base_llm=None)
    assert rec["status"] in ("FAILED", "PARTIAL") and rec["failures"]             # the run does not silently "succeed" without LLM answers
    assert rec["llm"]["cache"] == {"hits": 0, "misses": 0} and rec["report"]["findings"] == []


def test_run_tasks_writes_one_json_per_run_and_resumes(tmp_path, cassettes):
    c = cfg(tmp_path, cassettes)
    ts = [task("t1"), task("t2", "Compare LangGraph and CrewAI please")]
    recs = run_tasks(["C", "D"], ts, c, FakeRouter(), base_llm=DemoLLM(), workers=2)
    assert len(recs) == 4
    for s in ("C", "D"):
        for t in ("t1", "t2"):
            assert (tmp_path / "runs" / s / f"{t}.json").exists() and load_run(c, s, t)["task_id"] == t
    again = run_tasks(["C", "D"], ts, c, FakeRouter(), base_llm=None, workers=2)       # nothing re-run, no LLM needed
    assert [r["task_id"] for r in again] == [r["task_id"] for r in recs] or len(again) == 4
