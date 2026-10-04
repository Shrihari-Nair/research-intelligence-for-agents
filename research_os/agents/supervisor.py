"""Supervisor: decides WHICH workflow (never how to research). DIRECT mode makes zero LLM calls."""
from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, create_model

from research_os.contracts import FailureType, RouterOutput, RoutingMode
from research_os.graph.templates import load_workflows
from research_os.llm.models import model_for_role
from research_os.llm.structured import MalformedLLMOutput, complete_structured
from research_os.prompts import load_prompt

log = logging.getLogger(__name__)
DEFAULT_WORKFLOW = "general_research"
_RECALL = re.compile(r"\b(previously|earlier|last time|before)\b.{0,40}\b(find|found|research\w*|learn\w*|discuss\w*|conclude\w*)\b|"
                     r"\bwhat (did|do) we (find|learn|know|research|conclude)\b|\bhave we (already )?(researched|looked|covered)\b|"
                     r"\bremind me what we (found|learned)\b", re.I)


def is_recall_request(text: str) -> bool:
    """True for 'what did we previously find about X' style requests (answered from memory, no new router intent)."""
    return bool(_RECALL.search(text))


class SupervisorDecision(BaseModel):
    """The supervisor's output."""

    workflow: str
    intent: str
    mode: RoutingMode
    rationale: str
    llm_used: bool
    recall: list[dict] | None = Field(default=None, description="memory hits when the request asked about earlier research")
    fallback_reason: str | None = Field(default=None, description="FailureType name when a safe fallback was used")


class Supervisor:
    """Maps a RouterOutput to a workflow template (DIRECT) or asks the LLM (LLM_FALLBACK)."""

    def __init__(self, workflows_path: str = "configs/workflows.yaml", include_guess: bool = True):
        self.include_guess = include_guess
        self.workflows = load_workflows(workflows_path)
        names = tuple(self.workflows)
        self._choice_cls = create_model("SupervisorChoice", workflow=(Literal[names], ...), rationale=(str, ...))
        self._prompt = load_prompt("supervisor")

    def _decision(self, workflow: str, mode: RoutingMode, rationale: str, llm_used: bool,
                  fallback: FailureType | None = None) -> SupervisorDecision:
        return SupervisorDecision(workflow=workflow, intent=self.workflows[workflow]["intent"], mode=mode,
                                  rationale=rationale, llm_used=llm_used,
                                  fallback_reason=fallback.value if fallback else None)

    def recall(self, request: str, ctx) -> list[dict] | None:
        """Memory hits (with a stored report) for a recall request; None when memory has nothing or this is not a recall request."""
        if getattr(ctx, "memory", None) is None or not is_recall_request(request):
            return None
        from research_os.memory import memory_cfg
        cfg = memory_cfg()["recall"]
        res = ctx.tools.call("memory_search", {"query": request, "kinds": ["questions"], "k": cfg["k"],
                                               "min_similarity": cfg["min_similarity"]}, ctx, agent="supervisor")
        hits = [h for h in res.data.get("results", []) if h.get("report")]
        return hits or None

    def decide(self, request: str, routed: RouterOutput, ctx) -> SupervisorDecision:
        """Pick the workflow for this request."""
        ctx.events.emit("supervisor", "started", {"mode": routed.routing_mode.value, "router_intent": routed.intent,
                                                  "confidence": routed.confidence})
        recalled = self.recall(request, ctx)
        if recalled:
            d = self._decision(DEFAULT_WORKFLOW, routed.routing_mode, "request asks about earlier research; answered from memory", False)
            d.recall = recalled
            ctx.events.emit("supervisor", "finished", {"workflow": d.workflow, "intent": d.intent, "llm_used": False,
                                                       "fallback_reason": None, "recall_hits": len(recalled)})
            return d
        if routed.routing_mode == RoutingMode.DIRECT:
            if routed.workflow in self.workflows:
                d = self._decision(routed.workflow, routed.routing_mode, "router confident; template chosen by intent", False)
            else:
                d = self._decision(DEFAULT_WORKFLOW, routed.routing_mode,
                                   f"router named unknown workflow {routed.workflow!r}", False, FailureType.IMPOSSIBLE_PLAN)
        else:
            d = self._ask_llm(request, routed, ctx)
        ctx.events.emit("supervisor", "finished", {"workflow": d.workflow, "intent": d.intent, "llm_used": d.llm_used,
                                                   "fallback_reason": d.fallback_reason})
        return d

    def _ask_llm(self, request: str, routed: RouterOutput, ctx) -> SupervisorDecision:
        listing = "\n".join(f"- {n}: {w['description']}" for n, w in self.workflows.items())
        guess, conf = (routed.workflow, f"{routed.confidence:.2f}") if self.include_guess else ("none (no classifier available)", "n/a")
        prompt = self._prompt.render(workflows=listing, guess=guess, confidence=conf, request=request)
        try:
            choice, _ = complete_structured(ctx.llm, prompt, self._choice_cls, model=model_for_role("supervisor"),
                                            max_tokens=300, meta={"prompt": self._prompt.id})
        except MalformedLLMOutput as e:
            ctx.events.emit("supervisor", "failed", {"failure_type": FailureType.MALFORMED_LLM_OUTPUT.value, "error": str(e)})
            return self._decision(DEFAULT_WORKFLOW, routed.routing_mode, "LLM output invalid; defaulted", True,
                                  FailureType.MALFORMED_LLM_OUTPUT)
        except Exception as e:  # API failure must not take the run down
            log.warning("supervisor LLM call failed: %s", e)
            ctx.events.emit("supervisor", "failed", {"failure_type": FailureType.AGENT_FAILURE.value,
                                                     "error": f"{type(e).__name__}: {e}"})
            return self._decision(DEFAULT_WORKFLOW, routed.routing_mode, "LLM call failed; defaulted", True,
                                  FailureType.AGENT_FAILURE)
        return self._decision(choice.workflow, routed.routing_mode, choice.rationale, True)
