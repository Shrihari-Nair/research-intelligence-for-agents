"""Synthesis Agent: findings from VERIFIED claims only; every finding cites claim ids."""
from __future__ import annotations

from pydantic import BaseModel, Field

from research_os.agents.base import agent_cfg, merged
from research_os.agents.text import ShortIds
from research_os.contracts import ResearchStep
from research_os.harness.verification import ungrounded_numbers
from research_os.llm.models import model_for_role
from research_os.llm.structured import complete_structured
from research_os.prompts import load_prompt


class Finding(BaseModel):
    """One cited finding."""

    statement: str = Field(min_length=1)
    claim_ids: list[str]


class SynthesisDraft(BaseModel):
    """LLM output schema."""

    summary: str
    findings: list[Finding]


class SynthesisAgent:
    """Only claims with verdict SUPPORTED are visible to the model. No supported claims -> no LLM call."""

    name = "synthesis_agent"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or agent_cfg())["synthesis"]
        self.prompt = load_prompt("synthesis")

    def run(self, step: ResearchStep, inputs: dict, ctx, request: str) -> dict:
        supported = [c for c in merged_claims(inputs) if c["verdict"] == "SUPPORTED"]
        if not supported:
            return {"summary": "No verified findings were available for this request.", "findings": [], "claim_ids": []}
        sids = ShortIds([c["claim_id"] for c in supported], "c")      # short ids: the model must not copy 12-char hashes

        by_id = {c["claim_id"]: c for c in supported}
        evid = lambda c: [c["claim"], c["evidence_quote"]]  # noqa: E731
        everything = [request, str(len(supported)), *(t for c in supported for t in evid(c))]   # the number of claims is a fact about the list

        def check(d: SynthesisDraft) -> None:
            bad = ungrounded_numbers(d.summary, everything)
            if bad:
                raise ValueError(f"the summary contains number(s) {bad} that are not in the claims; use only numbers that appear in them")
            for f in d.findings:
                if not f.claim_ids:
                    raise ValueError("every finding must cite at least one claim id")
                unknown = [i for i in f.claim_ids if sids.real(i) is None]
                if unknown:
                    raise ValueError(f"unknown claim ids {unknown}; use only ids from the list (c1..c{len(sids)})")
                cited = [t for i in f.claim_ids for t in evid(by_id[sids.real(i)])]
                bad = ungrounded_numbers(f.statement, [request, *cited])
                if bad:
                    raise ValueError(f"finding {f.statement!r} contains number(s) {bad} that are not in the claims it cites")

        listing = "\n".join(f"{sids.short(c['claim_id'])}: {c['claim']}" for c in supported)
        hint = f"\nA previous attempt failed: {step.params['last_error']}. Fix this in your answer." if step.params.get("last_error") else ""
        prompt = self.prompt.render(request=step.params.get("request") or request, objective=step.objective + hint, claims=listing)
        draft, _ = complete_structured(ctx.llm, prompt, SynthesisDraft, model=model_for_role("synthesis"), max_tokens=self.cfg["max_output_tokens"],
                                       meta={"prompt": self.prompt.id}, validate=check)
        findings = [{"statement": f.statement, "claim_ids": [sids.real(i) for i in f.claim_ids]} for f in draft.findings[: self.cfg["max_findings"]]]
        return {"summary": draft.summary, "findings": findings, "claim_ids": sorted({i for f in findings for i in f["claim_ids"]})}


def merged_claims(inputs: dict) -> list[dict]:
    """All claim dicts from dependency outputs (deduplicated by claim_id, later outputs win)."""
    by_id: dict[str, dict] = {}
    for c in merged(inputs, "claims"):
        by_id[c["claim_id"]] = c
    return list(by_id.values())
