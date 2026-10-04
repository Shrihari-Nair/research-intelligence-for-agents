from research_os.router.entities import EntityExtractor
from research_os.router.evaluation.benchmark_v2 import benchmark_router_v2
from research_os.router.inference import ResearchRouter
from research_os.router.workflows import workflow_map

INTENTS = ["FACT_LOOKUP", "TECHNOLOGY_COMPARISON", "LITERATURE_REVIEW", "TECHNICAL_DEEP_DIVE",
           "COMPETITIVE_INTELLIGENCE", "MARKET_RESEARCH", "RESEARCH_MONITORING", "GENERAL_RESEARCH"]


class Fake:
    def predict(self, text):
        probs = {i: 0.01 for i in INTENTS}
        probs["FACT_LOOKUP"] = 0.93
        return {"intent": "FACT_LOOKUP", "confidence": 0.93, "probs": probs}


def test_benchmark_reports_predict_only_and_full_route_percentiles_and_entity_overhead():
    router = ResearchRouter(Fake(), EntityExtractor.from_config(), 0.6, workflow_map())
    res = benchmark_router_v2(router, ["Compare LangGraph and CrewAI", "what license does fastapi use"], warmup=3, runs=20)
    for key in ("predict_only", "full_route", "entities_only"):
        assert set(res[key]) >= {"p50_ms", "p95_ms", "p99_ms", "mean_ms"} and res[key]["p50_ms"] >= 0
    assert res["runs"] == 20 and "p95_overhead_ms" in res
