"""Planner: instantiate the workflow template with the request's entities, validate deterministically, estimate cost."""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from research_os.contracts import CostEstimate, ExecutionBudget, FailureType, ResearchPlan, ResearchStep
from research_os.graph.templates import AGENTS, budget_for, get_workflow, planner_cfg
from research_os.llm.models import model_for_role
from research_os.llm.structured import MalformedLLMOutput, complete_structured
from research_os.prompts import load_prompt
from research_os.tools import SEARCH_TOOLS

log = logging.getLogger(__name__)
_ID = re.compile(r"^[a-z0-9][a-z0-9_\-:]{0,40}$")
RETRIEVAL_AGENTS = ("search_agent", "source_agent")


class PlanValidationError(Exception):
    """The plan cannot be executed (failure type IMPOSSIBLE_PLAN)."""

    failure_type = FailureType.IMPOSSIBLE_PLAN


class ExtraStep(BaseModel):
    """An LLM-proposed retrieval step."""

    id: str
    objective: str
    agent: str
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class Refinement(BaseModel):
    """Schema of the optional refinement call."""

    extra_steps: list[ExtraStep] = Field(default_factory=list)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:30] or "x"


def validate_plan(plan: ResearchPlan, budget: ExecutionBudget, tool_names: list[str]) -> None:
    """Deterministic checks. Raises PlanValidationError listing every problem found."""
    problems: list[str] = []
    steps = plan.steps
    if not steps:
        raise PlanValidationError("plan has no steps")
    ids = [s.id for s in steps]
    if len(ids) != len(set(ids)):
        problems.append("duplicate step id(s)")
    known = set(ids)
    for s in steps:
        for d in s.depends_on:
            if d not in known:
                problems.append(f"step {s.id}: unknown dependency {d!r}")
        if s.agent not in AGENTS:
            problems.append(f"step {s.id}: unknown agent {s.agent!r}")
        for t in s.tools:
            if t not in tool_names:
                problems.append(f"step {s.id}: unknown tool {t!r}")
    if len(steps) > budget.max_steps:
        problems.append(f"plan has {len(steps)} steps, exceeds max_steps={budget.max_steps}")
    # cycle detection (Kahn)
    indeg = {s.id: len([d for d in s.depends_on if d in known]) for s in steps}
    ready = [i for i, n in indeg.items() if n == 0]
    seen = 0
    while ready:
        cur = ready.pop()
        seen += 1
        for s in steps:
            if cur in s.depends_on and s.id in indeg:
                indeg[s.id] -= 1
                if indeg[s.id] == 0:
                    ready.append(s.id)
    if seen != len(known):
        problems.append("dependency cycle")
    final = steps[-1]
    if final.agent != "synthesis_agent" or sum(s.agent == "synthesis_agent" for s in steps) != 1:
        problems.append("final step must be the only synthesis_agent step")
    elif not problems:
        ancestors: set[str] = set()
        stack = list(final.depends_on)
        by = {s.id: s for s in steps}
        while stack:
            cur = stack.pop()
            if cur not in ancestors:
                ancestors.add(cur)
                stack.extend(by[cur].depends_on)
        orphans = [i for i in ids if i != final.id and i not in ancestors]
        if orphans:
            problems.append(f"orphan step(s) not feeding synthesis: {orphans}")
    if problems:
        raise PlanValidationError("; ".join(problems))


def estimate_cost(plan: ResearchPlan, model: dict) -> CostEstimate:
    """Searches, expected sources, tool and LLM calls derived from the steps and the configured cost model."""
    searches = sum(sum(t in SEARCH_TOOLS for t in s.tools) for s in plan.steps)
    fetches = sum(model["fetches_per_read_step"] for s in plan.steps if "fetch_url" in s.tools)
    llm = sum(model["llm_calls_per_step"].get(s.agent, 0) for s in plan.steps)
    return CostEstimate(searches=searches, sources=searches * model["results_per_search"],
                        llm_calls=llm, tool_calls=searches + fetches)


class Planner:
    """Template instantiation (+ optional bounded LLM refinement) -> validated ResearchPlan."""

    def __init__(self, refine: bool | None = None, workflows_path: str = "configs/workflows.yaml"):
        self.cfg = planner_cfg(workflows_path)
        self.refine = self.cfg["refine"] if refine is None else refine
        self.workflows_path = workflows_path

    # ---- template instantiation ----
    def _instantiate(self, request: str, spec: dict, entities: dict, budget: ExecutionBudget, workflow: str):
        templates = spec["steps"]
        per_keys = {t["per"] for t in templates if t.get("per")}
        ents = {k: list(dict.fromkeys(entities.get(k, []))) for k in per_keys}
        notes: list[str] = []
        fixed = sum(1 for t in templates if not t.get("per"))
        per_templates = {k: sum(1 for t in templates if t.get("per") == k) for k in per_keys}

        def total() -> int:
            return fixed + sum(per_templates[k] * max(len(ents[k]), 1) for k in per_keys)

        while total() > budget.max_steps and any(len(v) > 1 for v in ents.values()):
            k = max((k for k in ents if len(ents[k]) > 1), key=lambda k: per_templates[k] * len(ents[k]))
            dropped = ents[k].pop()
            notes.append(f"entities truncated to fit max_steps={budget.max_steps}: dropped {dropped!r}")
        instances: dict[str, list[tuple[str, str]]] = {}  # template id -> [(step id, entity)]
        steps: list[ResearchStep] = []
        for t in templates:
            key = t.get("per")
            if key and ents[key]:
                pairs = [(f"{t['id']}:{_slug(e)}", e) for e in ents[key]]
            else:
                pairs = [(t["id"], request[:80])]
            instances[t["id"]] = pairs
            for sid, ent in pairs:
                deps: list[str] = []
                for d in t.get("depends_on", []):
                    d_pairs = instances[d]
                    d_per = next(x.get("per") for x in templates if x["id"] == d)
                    if d_per and key == d_per and ents[key]:
                        deps.append(next(i for i, e in d_pairs if e == ent))
                    else:
                        deps.extend(i for i, _ in d_pairs)
                steps.append(ResearchStep(id=sid, objective=t["objective"].format(request=request, entity=ent),
                                          agent=t["agent"], tools=list(t.get("tools", [])), depends_on=deps,
                                          params={"entity": ent if key and ents[key] else None, "request": request,
                                                  "recent": bool(spec.get("recent", False)), "workflow": workflow,
                                                  "allowed_source_types": spec.get("allowed_source_types", []),
                                                  "require_dates": bool(spec.get("require_dates", False))}))
        return steps, notes

    # ---- optional refinement ----
    def _refine(self, plan: ResearchPlan, request: str, ctx, tool_names: list[str]) -> ResearchPlan:
        prompt_t = load_prompt("planner")
        listing = "\n".join(f"- {s.id} ({s.agent}): {s.objective}" for s in plan.steps)
        prompt = prompt_t.render(workflow=plan.workflow, max_extra=str(self.cfg["max_extra_steps"]), steps=listing,
                                 tools=", ".join(tool_names), request=request)
        try:
            ref, _ = complete_structured(ctx.llm, prompt, Refinement, model=model_for_role("planner"), max_tokens=800,
                                         meta={"prompt": prompt_t.id})
        except (MalformedLLMOutput, Exception) as e:  # noqa: BLE001 - refinement is optional; never fail the run
            plan.notes.append(f"refinement ignored: {type(e).__name__}")
            return plan
        existing = {s.id for s in plan.steps}
        extras = ref.extra_steps[: self.cfg["max_extra_steps"]]
        for x in extras:
            bad = (x.agent not in RETRIEVAL_AGENTS or not _ID.match(x.id) or x.id in existing
                   or any(t not in tool_names for t in x.tools) or any(d not in existing for d in x.depends_on))
            if bad:
                plan.notes.append(f"refinement ignored: invalid extra step {x.id!r}")
                return plan
        new = [ResearchStep(id=x.id, objective=x.objective, agent=x.agent, tools=x.tools, depends_on=x.depends_on) for x in extras]
        if not new:
            return plan
        feeds = [s for s in plan.steps if s.agent == "source_agent" and any(
            d in {p.id for p in plan.steps if p.agent == "search_agent"} for d in s.depends_on)]
        for s in feeds or [plan.steps[-1]]:
            s.depends_on = [*s.depends_on, *[n.id for n in new]]
        plan.steps = [*plan.steps[:1], *new, *plan.steps[1:]] if feeds else [*plan.steps[:-1], *new, plan.steps[-1]]
        plan.notes.append(f"refinement added {len(new)} step(s): {[n.id for n in new]}")
        return plan

    def replan(self, request: str, decision, routed, ctx, reason: FailureType) -> ResearchPlan:
        """A second plan after recovery asked for one: the safe general_research template if the plan was invalid, otherwise the same
        workflow with broadened searches."""
        ctx.events.emit("planner", "info", {"replan": reason.value})
        if reason == FailureType.IMPOSSIBLE_PLAN:
            fallback = decision.model_copy(update={"workflow": "general_research", "intent": "GENERAL_RESEARCH"})
            plan = self.plan(request, fallback, routed.model_copy(update={"entities": {**routed.entities, "technologies": [], "companies": []}}), ctx)
            plan.notes.append("replanned: fell back to the general_research template after an invalid plan")
            return plan
        plan = self.plan(request, decision, routed, ctx)
        for s in plan.steps:
            if s.agent == "search_agent":
                s.params["broaden_level"] = 1
        plan.notes.append(f"replanned after {reason.value}: searches broadened")
        return plan

    def plan(self, request: str, decision, routed, ctx) -> ResearchPlan:
        """Build, (optionally) refine, cost and validate the plan. Raises PlanValidationError."""
        ctx.events.emit("planner", "started", {"workflow": decision.workflow})
        spec, budget = get_workflow(decision.workflow, self.workflows_path), budget_for(decision.workflow, self.workflows_path)
        steps, notes = self._instantiate(request, spec, routed.entities, budget, decision.workflow)
        plan = ResearchPlan(plan_id=f"plan-{ctx.run_id}", workflow=decision.workflow, steps=steps,
                            estimated_cost=CostEstimate(), notes=notes)
        tool_names = ctx.tools.names()
        if self.refine and ctx.llm is not None:
            plan = self._refine(plan, request, ctx, tool_names)
        plan.estimated_cost = estimate_cost(plan, self.cfg["cost_model"])
        try:
            validate_plan(plan, budget, tool_names)
        except PlanValidationError as e:
            ctx.events.emit("planner", "failed", {"failure_type": e.failure_type.value, "error": str(e)})
            raise
        ctx.events.emit("planner", "finished", {
            "steps": len(plan.steps), "cost": plan.estimated_cost.model_dump(), "notes": plan.notes,
            "plan": [{"id": s.id, "agent": s.agent, "objective": s.objective, "depends_on": s.depends_on, "tools": s.tools}
                     for s in plan.steps]})
        return plan
