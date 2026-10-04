"""Research agents. `build_agents()` returns the name -> agent mapping the executor uses."""
from __future__ import annotations

from research_os.agents.base import agent_cfg
from research_os.agents.evidence_agent import EvidenceAgent
from research_os.agents.search_agent import SearchAgent
from research_os.agents.source_agent import SourceAgent
from research_os.agents.synthesis_agent import SynthesisAgent
from research_os.agents.verifier import Verifier


def build_agents(cfg: dict | None = None) -> dict:
    """All step agents incl. the verifier."""
    cfg = cfg or agent_cfg()
    return {a.name: a for a in (SearchAgent(cfg), SourceAgent(cfg), EvidenceAgent(cfg), Verifier(cfg), SynthesisAgent(cfg))}
