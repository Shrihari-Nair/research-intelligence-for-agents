"""Evidence Agent: passages -> Claims with verbatim quotes (LLM extraction + deterministic quote validation)."""
from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from research_os.agents.base import AgentFailure, agent_cfg, merged
from research_os.agents.text import ShortIds, find_quote, normalize_ws
from research_os.contracts import Claim, FailureType, ResearchStep
from research_os.llm.models import model_for_role
from research_os.llm.structured import complete_structured
from research_os.prompts import load_prompt


class ClaimItem(BaseModel):
    """One LLM-proposed claim."""

    claim: str = Field(min_length=1)
    source_id: str
    evidence_quote: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ClaimBatch(BaseModel):
    """LLM output schema."""

    claims: list[ClaimItem] = Field(default_factory=list)


def _claim_id(source_id: str, quote: str, claim: str) -> str:
    return hashlib.sha256(f"{source_id}|{normalize_ws(quote)}|{claim}".encode()).hexdigest()[:12]


class EvidenceAgent:
    """Rejects any claim whose quote is not a verbatim substring of the stored source (and counts rejections)."""

    name = "evidence_agent"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or agent_cfg())["evidence"]
        self.prompt = load_prompt("evidence")

    def _batches(self, passages: list[dict]) -> list[list[dict]]:
        batches, cur, size = [], [], 0
        for p in passages:
            if cur and size + len(p["text"]) > self.cfg["batch_chars"]:
                batches.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += len(p["text"])
        if cur:
            batches.append(cur)
        return batches[: self.cfg["max_batches"]]

    def run(self, step: ResearchStep, inputs: dict, ctx, request: str) -> dict:
        passages = merged(inputs, "passages")
        if not passages:
            raise AgentFailure(FailureType.NO_USEFUL_SOURCES, "no passages to extract claims from")
        claims: list[Claim] = []
        seen: set[tuple[str, str]] = set()
        proposed = rejected = 0
        for batch in self._batches(passages):
            sids = ShortIds([p["source_id"] for p in batch], "s")        # short ids: the model must not copy 16-char hashes
            block = "\n\n".join(f"<<<SOURCE id={sids.short(p['source_id'])}\n{p['text']}\nSOURCE>>>" for p in batch)
            hint = f"\nA previous attempt failed: {step.params['last_error']}. Fix this in your answer." if step.params.get("last_error") else ""
            prompt = self.prompt.render(objective=step.objective + hint, passages=block, max_quote_chars=str(self.cfg["max_quote_chars"]),
                                        max_claims=str(self.cfg["max_claims_per_batch"]))
            parsed, _ = complete_structured(ctx.llm, prompt, ClaimBatch, model=model_for_role("evidence"),
                                            max_tokens=self.cfg["max_output_tokens"], meta={"prompt": self.prompt.id})
            for it in parsed.claims[: self.cfg["max_claims_per_batch"]]:
                proposed += 1
                real_sid = sids.real(it.source_id)
                src = ctx.sources.get(real_sid) if real_sid else None
                span = find_quote(it.evidence_quote, src.content) if src else None
                if span is None or len(it.evidence_quote) > self.cfg["max_quote_chars"] * 2:
                    rejected += 1
                    ctx.events.emit(f"step:{step.id}", "info", {"rejected claim": it.claim[:80], "reason": "quote not verbatim in source"})
                    continue
                key = (real_sid, normalize_ws(it.evidence_quote))
                if key in seen:
                    continue
                seen.add(key)
                claims.append(Claim(claim_id=_claim_id(real_sid, it.evidence_quote, it.claim), claim=it.claim.strip(),
                                    source_id=real_sid, evidence_quote=src.content[span[0]:span[1]],
                                    extractor_confidence=it.confidence))
        return {"claims": [c.model_dump(mode="json") for c in claims], "rejected_quotes": rejected, "claims_proposed": proposed}
