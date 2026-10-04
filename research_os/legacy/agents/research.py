"""Medical research agent (canned search tools by default)."""
from __future__ import annotations

from research_os.legacy.agents.base import AgentResult, phrase


class ResearchAgent:
    """Searches the (mock) literature and summarises titles; no medical advice."""

    name = "research_agent"

    def run(self, text: str, toolbox, llm, intent: str = "MEDICAL_RESEARCH") -> AgentResult:
        calls = [{"name": "search_pubmed", "args": {"query": text}}]
        res = toolbox.search_pubmed(text)["results"]
        if not res:
            calls.append({"name": "search_web", "args": {"query": text}})
            res = toolbox.search_web(text)["results"]
        titles = "; ".join(f"{r['title']} ({r['source']})" for r in res) or "no results"
        draft = f"Here is what I found: {titles}. This is general information, not medical advice."
        return {"agent": self.name, "tool_calls": calls, "response": phrase(llm, draft)}
