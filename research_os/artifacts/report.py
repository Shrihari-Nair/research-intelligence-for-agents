"""ResearchReport: the canonical, citation-checked artifact of a run. Limitations are generated from run data, never by an LLM."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FindingOut(BaseModel):
    """A finding and the claims that back it."""

    statement: str
    claim_ids: list[str]


class Evidence(BaseModel):
    """A SUPPORTED claim with its verbatim quote."""

    claim_id: str
    claim: str
    source_id: str
    evidence_quote: str
    verdict: Literal["SUPPORTED"] = "SUPPORTED"
    rationale: str | None = None


class SourceRef(BaseModel):
    """A source used in the run."""

    source_id: str
    title: str
    url: str
    source_type: str
    published_at: str | None = None


def check_integrity(report: ResearchReport) -> None:
    """Raise ValueError('citation integrity: ...') if any finding lacks a claim id or any claim id lacks evidence or a source."""
    ev = {e.claim_id: e for e in report.evidence}
    src = {s.source_id for s in report.sources}
    problems: list[str] = []
    for f in report.findings:
        if not f.claim_ids:
            problems.append(f"finding without a claim id: {f.statement[:60]!r}")
        for cid in f.claim_ids:
            if cid not in ev:
                problems.append(f"finding cites unknown claim id {cid!r}")
    for e in report.evidence:
        if e.source_id not in src:
            problems.append(f"claim {e.claim_id} has no source {e.source_id!r} in the sources list")
    for conflict in report.contradictions:
        for side in conflict.get("sides", []):
            for c in side.get("claims", []):
                if c.get("source_id") not in src:
                    problems.append(f"contradiction claim {c.get('claim_id')} has no source in the sources list")
    if problems:
        raise ValueError("citation integrity: " + "; ".join(problems))


class ResearchReport(BaseModel):
    """What the user receives. Every finding cites claim ids; every claim cites a source and a verbatim quote."""

    schema_version: str = "1"
    question: str
    status: Literal["SUCCESS", "PARTIAL", "FAILED", "CANCELLED"]
    executive_summary: str
    methodology: dict[str, Any] = Field(default_factory=dict)
    findings: list[FindingOut] = Field(default_factory=list)
    comparison: list[dict[str, Any]] | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _integrity(self) -> ResearchReport:
        check_integrity(self)
        return self


def generate_limitations(d: dict) -> list[str]:
    """Limitations derived from counts and flags of the run (deterministic text)."""
    out = ["Verification is automated (deterministic grounding checks plus an LLM judge); it is not a human review.",
           "Findings reflect the sources retrieved during this run and may be out of date or incomplete."]
    if d.get("llm_backend") == "demo":
        out.append("This report was produced with the offline demo LLM (extractive, not a language model); treat it as a pipeline demonstration.")
    if d.get("status") == "PARTIAL":
        why = d.get("halted_reason") or (f"steps that failed: {', '.join(d['failed_steps'])}" if d.get("failed_steps") else "some steps did not complete")
        out.append(f"This report is partial: {why}.")
    elif d.get("failed_steps"):
        out.append(f"Steps that failed or were skipped: {', '.join(d['failed_steps'])}.")
    if d.get("failed_steps") and d.get("status") == "PARTIAL":
        out.append(f"Steps that failed: {', '.join(d['failed_steps'])}.")
    st = d.get("stats") or {}
    if st.get("unsupported"):
        det = st.get("deterministic_rejections", 0)
        out.append(f"{st['unsupported']} claim(s) were dropped as unsupported by their evidence ({det} failed deterministic grounding checks).")
    if st.get("contradicted"):
        out.append(f"{st['contradicted']} claim(s) are contradicted or in conflict and are shown only in the contradictions section.")
    if d.get("conflicts"):
        out.append(f"{d['conflicts']} conflict(s) between sources were detected; see the contradictions section.")
    if d.get("rejected_quotes"):
        out.append(f"{d['rejected_quotes']} proposed claim(s) were rejected because their quote was not verbatim in the source.")
    if d.get("undated_sources"):
        out.append(f"{d['undated_sources']} source(s) have no publication date.")
    if d.get("flagged_sources"):
        out.append(f"{d['flagged_sources']} source(s) contained instruction-like text that was excluded (possible prompt injection).")
    if d.get("recoveries"):
        out.append(f"{d['recoveries']} recovery action(s) were needed during the run (retries, alternate tools, broadened queries or replanning).")
    if d.get("ask_user"):
        out.append("The system could not find useful sources and asked for a rephrased or narrower request.")
    if st.get("supported", 0) == 0 and not d.get("recalled"):
        out.append("No claim passed verification, so there are no findings.")
    return out


_COMPARISON_WORKFLOWS = ("technology_comparison", "competitive_intelligence")


def build_report(request: str, state: dict, ctx, *, llm_backend: str | None = None, status: str | None = None,
                 error: str | None = None) -> ResearchReport:
    """Assemble the ResearchReport for any finished run (completed, partial, recalled or failed)."""
    backend = llm_backend or ctx.config.get("llm_backend") or "unknown"
    now = datetime.now(timezone.utc).isoformat()
    if state.get("recall"):
        best = state["recall"][0]
        old = best.get("report") or {}
        summary = old.get("executive_summary") or old.get("summary") or best.get("text", "")
        evidence = [Evidence(**e) for e in old.get("evidence", [])]
        sources = [SourceRef(**s) for s in old.get("sources", [])]
        ids = {e.claim_id for e in evidence}
        findings = [FindingOut(**f) for f in old.get("findings", []) if f.get("claim_ids") and all(c in ids for c in f["claim_ids"])]
        return ResearchReport(question=request, status="SUCCESS", executive_summary=summary, findings=findings, evidence=evidence, sources=sources,
                              methodology={"workflow": "memory_recall", "note": "answered from earlier research stored in memory"},
                              limitations=["Answered from earlier stored research; no new research was run, so it may be out of date.",
                                           *generate_limitations({"llm_backend": backend, "recalled": True})[:1]],
                              run_metadata={"run_id": ctx.run_id, "recalled_from": [h["run_id"] for h in state["recall"]],
                                            "generated_at": now, "llm_backend": backend})
    if not state:
        return ResearchReport(question=request, status=status or "FAILED", executive_summary=f"The run failed: {error or 'unknown error'}",
                              limitations=generate_limitations({"llm_backend": backend, "status": "FAILED", "stats": {}}),
                              run_metadata={"run_id": ctx.run_id, "generated_at": now, "llm_backend": backend, "error": error})
    plan, execn = state.get("plan"), state.get("execution")
    steps = plan.steps if plan is not None else []
    ver = next((s.outputs for s in steps if s.agent == "verifier" and s.outputs), {}) if state.get("status") != "CANCELLED" else {}
    synth = next((s.outputs for s in steps if s.agent == "synthesis_agent" and s.outputs), {})
    all_claims = ver.get("claims", [])
    supported = [c for c in all_claims if c.get("verdict") == "SUPPORTED"]
    ids = {c["claim_id"] for c in supported}
    findings = [FindingOut(**f) for f in synth.get("findings", []) if f.get("claim_ids") and all(c in ids for c in f["claim_ids"])]
    dropped_findings = len(synth.get("findings", [])) - len(findings)
    final_status = status or state.get("status", "FAILED")
    partial = state.get("partial_artifact") or {}
    summary = synth.get("summary") or partial.get("message") or ("The run did not produce a summary." if final_status != "SUCCESS" else "")
    if final_status == "CANCELLED":
        summary = f"The run was cancelled: {state.get('error') or 'no reason given'}."
        findings, dropped_findings = [], 0
    elif not findings and ver.get("conflicts") and final_status != "FAILED":
        summary = ("The sources disagree on this question. The conflicting claims are shown in the Contradictions section; "
                   "none of them is presented as a finding.")
    if state.get("error") and not synth.get("summary"):
        summary = f"{summary} ({state['error']})".strip()
    src_by_id = {s.source_id: s for s in ctx.sources.all()}
    sources = [SourceRef(source_id=s.source_id, title=s.title, url=s.url, source_type=s.source_type,
                         published_at=s.published_at.isoformat() if s.published_at else None) for s in src_by_id.values()]
    evidence = [Evidence(claim_id=c["claim_id"], claim=c["claim"], source_id=c["source_id"], evidence_quote=c["evidence_quote"],
                         rationale=c.get("verdict_rationale")) for c in supported if c["source_id"] in src_by_id]
    workflow = state["decision"].workflow if state.get("decision") else None
    comparison = None
    if workflow in _COMPARISON_WORKFLOWS:
        by_id = {c["claim_id"]: c for c in supported}
        producer: dict[str, str] = {}          # claim id -> entity of the step that extracted it (fallback attribution)
        order: list[str] = []
        for s in steps:
            ent = s.params.get("entity") if s.agent == "evidence_agent" else None
            if ent:
                order.append(ent)
                for c in (s.outputs or {}).get("claims", []):
                    producer.setdefault(c["claim_id"], ent)
        order = list(dict.fromkeys(order))
        rows: dict[str, list[dict]] = {e: [] for e in order}
        for cid, c in by_id.items():
            mentioned = [e for e in order if e.lower() in c["claim"].lower() or e.lower() in c["evidence_quote"].lower()]
            for e in mentioned or ([producer[cid]] if cid in producer else []):
                rows[e].append({"claim_id": cid, "statement": c["claim"]})
        comparison = [{"entity": e, "claims": cl} for e, cl in rows.items() if cl] or None
    stats = dict(ver.get("stats", {}))
    flagged = sum(len(s.outputs.get("flagged_sources", [])) for s in steps if s.agent == "source_agent" and s.outputs)
    rejected = sum(s.outputs.get("rejected_quotes", 0) for s in steps if s.agent == "evidence_agent" and s.outputs)
    failed = [s.id for s in steps if s.status in ("failed", "skipped")]
    limitations = generate_limitations({
        "status": final_status, "halted_reason": (execn.halted_reason if execn else None) or partial.get("halted_reason"),
        "failed_steps": failed, "stats": stats, "conflicts": len(ver.get("conflicts", [])), "rejected_quotes": rejected,
        "undated_sources": sum(s.published_at is None for s in src_by_id.values()), "flagged_sources": flagged,
        "recoveries": len(execn.recoveries) if execn else 0, "llm_backend": backend, "ask_user": execn.ask_user if execn else None})
    if dropped_findings:
        limitations.append(f"{dropped_findings} finding(s) were removed because they cited claims that did not pass verification.")
    tools_used = sorted({e.node[5:] for e in ctx.events.events if e.node.startswith("tool:") and e.type == "started"})
    ro, dec = state.get("router_output"), state.get("decision")
    methodology = {
        "workflow": workflow, "router_intent": ro.intent if ro else None, "router_confidence": ro.confidence if ro else None,
        "router_mode": ro.routing_mode.value if ro else None, "supervisor_llm_used": dec.llm_used if dec else None,
        "supervisor_fallback": dec.fallback_reason if dec else None,
        "plan": [{"id": s.id, "agent": s.agent, "status": s.status, "attempts": s.attempts} for s in steps],
        "agents": sorted({s.agent for s in steps}), "tools_used": tools_used, "plan_notes": plan.notes if plan is not None else [],
        "prior_findings_offered": len(state.get("prior_findings", []))}
    meta = {"run_id": ctx.run_id, "status": final_status, "llm_backend": backend, "budget": ctx.budget.snapshot(), "generated_at": now,
            "verification": {**{k: stats.get(k, 0) for k in ("supported", "contradicted", "unsupported", "total", "deterministic_rejections",
                                                              "judge_calls")}, "gate": ver.get("gate")},
            "recoveries": execn.recoveries if execn else [], "monitoring_diff": state.get("monitoring_diff")}
    return ResearchReport(question=request, status=final_status, executive_summary=summary, methodology=methodology, findings=findings,
                          comparison=comparison, evidence=evidence, contradictions=ver.get("conflicts", []), limitations=limitations,
                          sources=sources, run_metadata=meta)
