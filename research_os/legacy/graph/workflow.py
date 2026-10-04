"""START -> intent_router -> (by intent) -> agent node -> respond -> END."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from research_os.legacy.agents.booking import BookingAgent
from research_os.legacy.agents.emergency import EmergencyAgent
from research_os.legacy.agents.faq import FAQAgent
from research_os.legacy.agents.prescription import PrescriptionAgent
from research_os.legacy.agents.research import ResearchAgent
from research_os.legacy.graph.state import GraphState

INTENT_TO_AGENT = {
    "BOOK_APPOINTMENT": "booking_agent",
    "RESCHEDULE_APPOINTMENT": "booking_agent",
    "CANCEL_APPOINTMENT": "booking_agent",
    "PRESCRIPTION_QUERY": "prescription_agent",
    "FOLLOWUP_QUERY": "prescription_agent",
    "MEDICAL_RESEARCH": "research_agent",
    "GENERAL_QUERY": "faq_agent",
    "EMERGENCY": "emergency_agent",
}
AGENTS = {a.name: a for a in (BookingAgent(), PrescriptionAgent(), ResearchAgent(), EmergencyAgent(), FAQAgent())}


def build_graph(system, toolbox, llm):
    """Compile the workflow; `system` is any RoutingSystem (LLM-only, MiniLM-only or hybrid)."""

    def intent_router(state: GraphState) -> dict:
        return {"routing": dict(system.route(state["text"]))}

    def make_agent_node(agent):
        def node(state: GraphState) -> dict:
            out = agent.run(state["text"], toolbox, llm, state["routing"]["intent"])
            return {"agent_name": out["agent"], "tool_calls": out["tool_calls"], "response": out["response"]}
        return node

    def respond(state: GraphState) -> dict:
        return {"response": state["response"]}

    def pick(state: GraphState) -> str:
        return INTENT_TO_AGENT[state["routing"]["intent"]]

    g = StateGraph(GraphState)
    g.add_node("intent_router", intent_router)
    g.add_node("respond", respond)
    for name, agent in AGENTS.items():
        g.add_node(name, make_agent_node(agent))
        g.add_edge(name, "respond")
    g.add_edge(START, "intent_router")
    g.add_conditional_edges("intent_router", pick, {n: n for n in AGENTS})
    g.add_edge("respond", END)
    return g.compile()


def run_text(graph, text: str) -> GraphState:
    """Run one utterance through the compiled graph."""
    return graph.invoke({"text": text})
