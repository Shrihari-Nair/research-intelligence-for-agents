"use client";
import { useMemo, useState } from "react";
import type { ClaimView, Conflict, RunDetail } from "@/api/client";
import { SourceViewer } from "@/components/SourceViewer";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";

function ClaimRow({ c, active, onClick }: { c: ClaimView; active: boolean; onClick: () => void }) {
  return (
    <button data-testid="claim-row" onClick={onClick} className={cn("w-full rounded border p-2 text-left text-sm hover:bg-slate-50", active && "border-blue-500 bg-blue-50")}>
      <div className="flex items-start justify-between gap-2"><span>{c.claim}</span><StatusBadge status={c.verdict} /></div>
      {c.verdict_rationale && <div className="mt-1 text-xs text-slate-500">{c.verdict_rationale}</div>}
    </button>
  );
}

function ConflictCard({ k, onPick }: { k: Conflict; onPick: (id: string) => void }) {
  return (
    <div className="rounded border border-red-200 bg-red-50 p-3" data-testid="conflict-card">
      <div className="mb-2 text-sm font-semibold">Conflict: {k.subject}</div>
      <div className="grid gap-2 md:grid-cols-2">
        {k.sides.map((s) => (
          <div key={s.stance} className="rounded bg-white p-2 text-sm">
            <div className="mb-1 text-xs font-semibold uppercase text-slate-500">{s.stance}</div>
            {s.claims.map((c) => (
              <button key={c.claim_id} className="mb-1 block w-full text-left hover:underline" onClick={() => onPick(c.claim_id)}>
                {c.claim}<div className="text-xs italic text-slate-500">“{c.evidence_quote}”</div>
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function EvidenceExplorer({ detail, initialClaim }: { detail: RunDetail; initialClaim?: string }) {
  const claims = detail.claims as unknown as ClaimView[];
  const conflicts = detail.conflicts as unknown as Conflict[];
  const [sel, setSel] = useState<string | undefined>(initialClaim);
  const groups = useMemo(() => ({
    supported: claims.filter((c) => c.verdict === "SUPPORTED"),
    conflicting: claims.filter((c) => c.verdict === "CONTRADICTED"),
    dropped: claims.filter((c) => c.verdict === "UNSUPPORTED" || c.verdict === "UNVERIFIED"),
  }), [claims]);
  const selected = claims.find((c) => c.claim_id === sel);
  if (!claims.length) return <Card><CardBody className="text-sm text-slate-500">No claims were extracted for this run yet.</CardBody></Card>;
  const section = (title: string, list: ClaimView[], testid: string) => (
    <div data-testid={testid}>
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">{title} ({list.length})</div>
      <div className="space-y-1.5">{list.map((c) => <ClaimRow key={c.claim_id} c={c} active={c.claim_id === sel} onClick={() => setSel(c.claim_id)} />)}</div>
    </div>
  );
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        {conflicts.map((k) => <ConflictCard key={k.subject} k={k} onPick={setSel} />)}
        {section("Supported", groups.supported, "group-supported")}
        {groups.conflicting.length > 0 && section("Conflicting", groups.conflicting, "group-conflicting")}
        {groups.dropped.length > 0 && section("Dropped (not supported by their quote)", groups.dropped, "group-dropped")}
      </div>
      <Card className="h-fit lg:sticky lg:top-4">
        <CardHeader>Source and quote</CardHeader>
        <CardBody>
          {selected ? (
            <div className="space-y-2">
              <div className="text-sm"><b>Claim:</b> {selected.claim}</div>
              <div className="rounded bg-yellow-50 p-2 text-sm italic">“{selected.evidence_quote}”</div>
              <SourceViewer runId={detail.run_id} sourceId={selected.source_id} quotes={[selected.evidence_quote]} />
            </div>
          ) : <div className="text-sm text-slate-500">Select a claim to see its source with the quote highlighted.</div>}
        </CardBody>
      </Card>
    </div>
  );
}
