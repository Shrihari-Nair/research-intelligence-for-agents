import json

from research_os.agents.supervisor import Supervisor
from research_os.contracts import RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools.registry import ToolRegistry


class FakeRouter:
    def __init__(self, intent, workflow, mode, conf):
        self.args = (intent, workflow, mode, conf)

    def route(self, text):
        i, w, m, c = self.args
        return RouterOutput(intent=i, confidence=c, routing_mode=m, workflow=w, latency_ms=1.0,
                            entities={"technologies": ["LangGraph", "CrewAI"], "companies": [], "topics": [], "time_scope": []})


def test_router_then_supervisor_nodes_run_in_order_and_fill_state():
    ctx = RunContext.create("r", {}, llm=ScriptedLLM([]), tools=ToolRegistry())
    g = build_graph(FakeRouter("TECHNOLOGY_COMPARISON", "technology_comparison", RoutingMode.DIRECT, 0.9), Supervisor(), ctx,
                    stop_after="supervisor")
    out = g.invoke(initial_state("Compare LangGraph and CrewAI"))
    assert out["router_output"].intent == "TECHNOLOGY_COMPARISON"
    assert out["decision"].workflow == "technology_comparison" and out["decision"].llm_used is False
    nodes = [e.node for e in ctx.events.events if e.type == "started"]
    assert nodes.index("router") < nodes.index("supervisor")


def test_low_confidence_request_goes_through_the_llm_supervisor():
    llm = ScriptedLLM([json.dumps({"workflow": "market_research", "rationale": "asks about market size"})])
    ctx = RunContext.create("r", {}, llm=llm, tools=ToolRegistry())
    g = build_graph(FakeRouter("GENERAL_RESEARCH", "general_research", RoutingMode.LLM_FALLBACK, 0.3), Supervisor(), ctx,
                    stop_after="supervisor")
    out = g.invoke(initial_state("how big is the vector database market"))
    assert out["decision"].workflow == "market_research" and out["decision"].llm_used is True
