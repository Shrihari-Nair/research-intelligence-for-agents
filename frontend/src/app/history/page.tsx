"use client";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { listRuns, type RunSummary } from "@/api/client";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardBody } from "@/components/ui/card";

const FIELD = "rounded-xl border border-white/10 bg-ink-3 px-3 py-2 text-sm text-fg placeholder:text-fg-faint transition-colors hover:border-volt/40 focus:border-volt focus:outline-none";

export default function History() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  useEffect(() => { listRuns().then(setRuns).catch((e) => setErr(String(e))); }, []);
  const rows = useMemo(() => (runs ?? []).filter((r) => (!status || r.status === status) && (!q || r.request.toLowerCase().includes(q.toLowerCase()))), [runs, status, q]);
  if (err) return <div role="alert" className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-4 text-rose-300">{err}</div>;
  if (!runs) return (
    <div className="space-y-3" aria-busy>{[0, 1, 2, 3].map((i) => <div key={i} className="skeleton h-14 rounded-xl" />)}<span className="sr-only">Loading history…</span></div>
  );
  return (
    <main className="space-y-6">
      <div className="animate-fade-up">
        <div className="font-mono text-xs uppercase tracking-[0.25em] text-volt">archive</div>
        <h1 className="mt-2 text-5xl font-extrabold tracking-[-0.04em] sm:text-6xl">Run <span className="text-outline">history</span></h1>
      </div>
      <div className="flex flex-wrap items-center gap-3 animate-fade-up" style={{ animationDelay: "80ms" }}>
        <input data-testid="history-search" aria-label="Filter by text" className={`${FIELD} w-72`} placeholder="filter by text" value={q} onChange={(e) => setQ(e.target.value)} />
        <select data-testid="history-status" aria-label="Filter by status" className={FIELD} value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">all statuses</option>{["SUCCESS", "PARTIAL", "FAILED", "CANCELLED", "RUNNING", "AWAITING_APPROVAL"].map((s) => <option key={s}>{s}</option>)}
        </select>
        <span className="ml-auto font-mono text-xs text-fg-faint tnum">{rows.length} / {runs.length} runs</span>
      </div>
      <Card className="animate-fade-up" style={{ animationDelay: "160ms" }}><CardBody className="overflow-x-auto p-0">
        <table className="w-full text-sm" data-testid="history-table">
          <thead className="text-left font-mono text-[11px] uppercase tracking-[0.14em] text-fg-faint"><tr className="border-b border-line">
            {["Request", "Status", "Workflow", "Sources", "Verified claims", "Conflicts", "Duration"].map((h) => <th key={h} className="px-4 py-3 font-medium">{h}</th>)}</tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.run_id} className="group border-t border-line transition-colors hover:bg-volt/[0.04]" data-testid="history-row">
              <td className="max-w-md px-4 py-3.5"><Link className="font-medium text-fg transition-colors group-hover:text-volt" href={`/runs/${r.run_id}`}>{r.request}</Link></td>
              <td className="px-4 py-3.5"><StatusBadge status={r.status} /></td><td className="px-4 py-3.5 font-mono text-xs text-fg-dim">{r.workflow ?? "-"}</td>
              <td className="tnum px-4 py-3.5 font-mono">{r.n_sources}</td><td className="tnum px-4 py-3.5 font-mono">{r.n_supported_claims}</td><td className="tnum px-4 py-3.5 font-mono">{r.n_conflicts}</td>
              <td className="tnum px-4 py-3.5 font-mono text-fg-dim">{r.duration_s == null ? "-" : `${r.duration_s.toFixed(1)}s`}</td></tr>))}
            {rows.length === 0 && <tr><td colSpan={7} className="px-4 py-14 text-center text-fg-dim">No runs match.</td></tr>}</tbody>
        </table>
      </CardBody></Card>
    </main>
  );
}
