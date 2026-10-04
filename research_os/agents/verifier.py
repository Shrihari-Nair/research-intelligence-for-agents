"""Verifier: deterministic checks, then an LLM judge, then conflict detection and the PASS/FAIL gate."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from research_os.agents.base import agent_cfg
from research_os.agents.synthesis_agent import merged_claims
from research_os.agents.text import ShortIds
from research_os.contracts import ResearchStep
from research_os.graph.templates import get_workflow
from research_os.harness.verification import detect_conflicts, deterministic_checks, gate
from research_os.llm.models import model_for_role
from research_os.llm.structured import MalformedLLMOutput, complete_structured
from research_os.prompts import load_prompt


class Judgement(BaseModel):
    """One verdict from the judge."""

    claim_id: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
    rationale: str


class JudgeBatch(BaseModel):
    """Judge output schema."""

    judgements: list[Judgement]


class ConflictCheck(BaseModel):
    """One confirmation of a candidate conflict."""

    id: str
    subject_a: str = ""
    subject_b: str = ""
    same_subject: bool = True
    contradict: bool
    reason: str = ""


class ConflictChecks(BaseModel):
    """Conflict-confirmation output schema."""

    checks: list[ConflictCheck]


class Verifier:
    """No unsupported claim reaches synthesis: it only sees SUPPORTED claims."""

    name = "verifier"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or agent_cfg())["verifier"]
        self.prompt = load_prompt("verifier_judge")
        self.conflict_prompt = load_prompt("conflict_confirm")

    def _judge(self, batch: list[dict], sources: dict, ctx) -> dict[str, Judgement]:
        """Judge one batch. The model sees short ids (c1..cN); verdicts are mapped back to the real claim ids."""
        ids = ShortIds([c["claim_id"] for c in batch], "c")
        block = "\n\n".join(f"<<<CLAIM id={ids.short(c['claim_id'])}\nClaim: {c['claim']}\nEvidence quote: {c['evidence_quote']}\n"
                            f"Source type: {sources[c['source_id']].source_type}\nCLAIM>>>" for c in batch)
        want = {ids.short(c["claim_id"]).lower() for c in batch}

        def check(d: JudgeBatch) -> None:
            got = [j.claim_id.strip().lower() for j in d.judgements]
            if set(got) - want:
                raise ValueError(f"unknown claim ids {sorted(set(got) - want)}; use only {sorted(want)}")
            if want - set(got):
                raise ValueError(f"missing verdicts for claim ids {sorted(want - set(got))}")

        parsed, _ = complete_structured(ctx.llm, self.prompt.render(claims=block), JudgeBatch, model=model_for_role("verifier_judge"),
                                        max_tokens=2048, meta={"prompt": self.prompt.id}, validate=check)
        return {ids.real(j.claim_id): j for j in parsed.judgements}

    def _confirm_conflicts(self, candidates: list[dict], ctx) -> list[dict]:
        """Keep only the candidate conflicts the LLM confirms are about the same subject. A failed check confirms nothing."""
        cap = self.cfg.get("max_conflict_checks", 6)
        cands = candidates[:cap]
        ids = [f"k{i + 1}" for i in range(len(cands))]

        def block(i: str, k: dict) -> str:
            side = lambda sd: "\n".join(f"  - {c['claim']} (quote: \"{c['evidence_quote']}\"; source: {c['source_url']})" for c in sd["claims"])  # noqa: E731
            return f"<<<CANDIDATE id={i} about {k['subject']}\nSide A asserts:\n{side(k['sides'][0])}\nSide B negates:\n{side(k['sides'][1])}\nCANDIDATE>>>"

        def check(d: ConflictChecks) -> None:
            got = {c.id.strip().lower() for c in d.checks}
            if got != set(ids):
                raise ValueError(f"answer every candidate id exactly once: {ids}")

        try:
            parsed, _ = complete_structured(ctx.llm, self.conflict_prompt.render(candidates="\n\n".join(block(i, k) for i, k in zip(ids, cands, strict=True))),
                                            ConflictChecks, model=model_for_role("verifier_judge"), max_tokens=1024,
                                            meta={"prompt": self.conflict_prompt.id}, validate=check)
        except MalformedLLMOutput:
            return []
        yes = {c.id.strip().lower() for c in parsed.checks if c.contradict and c.same_subject}
        return [k for i, k in zip(ids, cands, strict=True) if i in yes]

    def run(self, step: ResearchStep, inputs: dict, ctx, request: str) -> dict:
        wf = get_workflow(step.params.get("workflow", "general_research"))
        sources = {s.source_id: s for s in ctx.sources.all()}
        claims = [dict(c) for c in merged_claims(inputs)]
        pending: list[dict] = []
        det_rejected = 0
        for c in claims:
            reasons = deterministic_checks(c, sources, wf)
            if reasons:
                c["verdict"], c["verdict_rationale"] = "UNSUPPORTED", "; ".join(reasons)
                c["verdict_source"] = "deterministic"
                det_rejected += 1
            else:
                pending.append(c)
        judge_calls = 0
        size = self.cfg["judge_batch_size"]
        for i in range(0, min(len(pending), size * self.cfg["max_judge_batches"]), size):
            batch = pending[i: i + size]
            verdicts = self._judge(batch, sources, ctx)
            judge_calls += 1
            for c in batch:
                j = verdicts[c["claim_id"]]
                c["verdict"], c["verdict_rationale"], c["verdict_source"] = j.verdict, j.rationale, "judge"
        for c in pending:  # beyond the judging cap: never trust unjudged claims
            if c["verdict"] == "UNVERIFIED":
                c["verdict"], c["verdict_rationale"] = "UNSUPPORTED", "not judged: verification batch limit reached"
                c["verdict_source"] = "unjudged"
        candidates = detect_conflicts([c for c in claims if c["verdict"] == "SUPPORTED"], sources)
        conflicts = self._confirm_conflicts(candidates, ctx) if candidates else []
        by_id = {c["claim_id"]: c for c in claims}
        for k in conflicts:
            ids = [x["claim_id"] for side in k["sides"] for x in side["claims"]]
            for cid in ids:
                by_id[cid]["verdict"], by_id[cid]["verdict_source"] = "CONTRADICTED", "conflict"
                by_id[cid]["verdict_rationale"] = f"conflict ({k['subject']}) with claim(s) {[i for i in ids if i != cid]} from another source"
        counts = {v: sum(c["verdict"] == v for c in claims) for v in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")}
        stypes = [sources[c["source_id"]].source_type for c in claims if c["verdict"] == "SUPPORTED"]
        g = gate(counts, stypes, wf, n_conflicts=len(conflicts),
                 n_conflict_claims=sum(c["verdict_source"] == "conflict" for c in claims))
        ctx.events.emit(f"step:{step.id}", "info", {"gate": g["result"], "counts": counts, "conflicts": len(conflicts)})
        return {"claims": claims, "conflicts": conflicts, "gate": g,
                "stats": {"total": len(claims), "supported": counts["SUPPORTED"], "contradicted": counts["CONTRADICTED"],
                          "unsupported": counts["UNSUPPORTED"], "deterministic_rejections": det_rejected, "judge_calls": judge_calls,
                          "conflict_candidates": len(candidates)}}
