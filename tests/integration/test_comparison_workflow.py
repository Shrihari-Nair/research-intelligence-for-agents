import json

from research_os.agents import build_agents
from research_os.agents.supervisor import Supervisor
from research_os.contracts import Claim, RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry

LG, CREW = "https://docs.example-agents.dev/langgraph/overview", "https://docs.example-agents.dev/crewai/overview"
LG_Q, CREW_Q = "It supports persistence through checkpointers", "CrewAI is a framework for orchestrating role-playing autonomous agents"


class Router:
    def route(self, text):
        return RouterOutput(intent="TECHNOLOGY_COMPARISON", confidence=0.95, routing_mode=RoutingMode.DIRECT,
                            workflow="technology_comparison", latency_ms=1,
                            entities={"technologies": ["LangGraph", "CrewAI"], "companies": [], "topics": [], "time_scope": []})


class MarkAllSupported:
    """Stand-in for the M12 verifier."""

    name = "verifier"

    def run(self, step, inputs, ctx, request):
        claims = [dict(c, verdict="SUPPORTED") for d in inputs.values() for c in d.get("claims", [])]
        return {"claims": claims}


class PromptAwareLLM:
    """Answers each pipeline prompt from its content (prompts use short ids), the way a competent model would."""

    QUOTES = {LG_Q: "LangGraph supports persistence", CREW_Q: "CrewAI orchestrates role-playing agents"}

    def __init__(self):
        self.calls = 0
        self.last_claims_prompt = ""
        self.done: set[str] = set()

    def complete(self, prompt, *, model=None, json_mode=False, max_tokens=1024, meta=None):
        import re

        from research_os.llm.client import LLMResponse
        self.calls += 1
        pid = (meta or {}).get("prompt", "")
        if pid.startswith("evidence"):
            blocks = re.findall(r"<<<SOURCE id=(\S+)\n(.*?)\nSOURCE>>>", prompt, re.S)
            out = []
            for quote, claim in self.QUOTES.items():                 # each distinct quote is extracted once across the run
                hit = next(((sid, text) for sid, text in blocks if quote in text), None)
                if hit and quote not in self.done:
                    self.done.add(quote)
                    out.append({"claim": claim, "source_id": hit[0], "evidence_quote": quote, "confidence": 0.9})
                    break
            return LLMResponse(text=json.dumps({"claims": out}))
        if pid.startswith("verifier_judge"):
            ids = re.findall(r"<<<CLAIM id=(\S+)\n", prompt)
            return LLMResponse(text=json.dumps({"judgements": [{"claim_id": i, "verdict": "SUPPORTED", "rationale": "direct"} for i in ids]}))
        if pid.startswith("synthesis"):
            self.last_claims_prompt = prompt
            listed = re.findall(r"^(c\d+): (.+)$", prompt, re.M)
            return LLMResponse(text=json.dumps({"summary": "Both are agent frameworks.",
                                                "findings": [{"statement": text, "claim_ids": [cid]} for cid, text in listed]}))
        return LLMResponse(text="{}")


def test_technology_comparison_runs_end_to_end_with_valid_quotes():
    llm = PromptAwareLLM()
    ctx = RunContext.create("run1", {}, llm=llm, tools=build_registry(mode="fixture"))
    out = build_graph(Router(), Supervisor(), ctx, agents=build_agents()).invoke(initial_state("Compare LangGraph and CrewAI"))   # real verifier + judge
    assert out["status"] == "SUCCESS"
    steps = {s.id: s for s in out["plan"].steps}
    assert set(steps) >= {"search:langgraph", "search:crewai", "extract:langgraph", "extract:crewai", "verify", "synthesize"}
    all_claims = [Claim(**c) for c in steps["verify"].outputs["claims"]]
    assert len(all_claims) == 2 and {c.verdict for c in all_claims} == {"SUPPORTED"}
    assert steps["verify"].outputs["gate"]["result"] == "PASS"
    for c in all_claims:                                           # hard rule 8: verbatim quote in the stored source
        assert c.evidence_quote in ctx.sources.get(c.source_id).content
    final = steps["synthesize"].outputs
    real_ids = {c.claim_id for c in all_claims}
    assert len(final["findings"]) == 2 and {f["claim_ids"][0] for f in final["findings"]} == real_ids   # short ids mapped back to real ones
    assert all(real_ids >= set(f["claim_ids"]) for f in final["findings"])
    assert "c1:" in llm.last_claims_prompt and not any(cid in llm.last_claims_prompt for cid in real_ids)   # the model never saw a hash
    assert ctx.budget.usage()["llm_calls"] == 4 and ctx.budget.usage()["searches"] >= 4
    kinds = {e.node for e in ctx.events.events}
    assert {"router", "supervisor", "planner", "step:verify"} <= kinds


def test_a_hostile_source_cannot_change_agent_behaviour():
    class HostileRouter(Router):
        def route(self, text):
            out = super().route(text)
            return out.model_copy(update={"entities": {**out.entities, "technologies": ["LangGraph"]}})
    seen = []

    class Spy(ScriptedLLM):
        def complete(self, prompt, **kw):
            seen.append(prompt)
            return super().complete(prompt, **kw)
    llm = Spy([json.dumps({"claims": []}), json.dumps({"claims": []})])
    ctx = RunContext.create("run2", {}, llm=llm, tools=build_registry(mode="fixture"))
    build_graph(HostileRouter(), Supervisor(), ctx, agents={**build_agents(), "verifier": MarkAllSupported()}).invoke(
        initial_state("LangGraph tips for beginners"))
    assert seen and all("IGNORE ALL PREVIOUS" not in p and "admin mode" not in p for p in seen)
