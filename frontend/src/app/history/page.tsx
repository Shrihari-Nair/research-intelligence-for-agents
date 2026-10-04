"use client";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { listRuns, type RunSummary } from "@/api/client";
import { StatusBadge } from "@/components/StatusBadge";
import { Card, CardBody } from "@/components/ui/card";

export default function History() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  useEffect(() => { listRuns().then(setRuns).catch((e) => setErr(String(e))); }, []);
  const rows = useMemo(() => (runs ?? []).filter((r) => (!status || r.status === status) && (!q || r.request.toLowerCase().includes(q.toLowerCase()))), [runs, status, q]);
  if (err) return <div className="text-red-700">{err}</div>;
  if (!runs) return <div className="text-slate-500">Loading history…</div>;
  return (
    <main className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">Run history</h1>
        <input data-testid="history-search" className="rounded border px-2 py-1 text-sm" placeholder="filter by text" value={q} onChange={(e) => setQ(e.target.value)} />
        <select data-testid="history-status" className="rounded border p-1 text-sm" value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">all statuses</option>{["SUCCESS", "PARTIAL", "FAILED", "CANCELLED", "RUNNING", "AWAITING_APPROVAL"].map((s) => <option key={s}>{s}</option>)}
        </select>
      </div>
      <Card><CardBody className="overflow-x-auto p-0">
        <table className="w-full text-sm" data-testid="history-table">
          <thead className="bg-slate-100 text-left text-xs uppercase text-slate-500"><tr>
            {["Request", "Status", "Workflow", "Sources", "Verified claims", "Conflicts", "Duration"].map((h) => <th key={h} className="px-3 py-2">{h}</th>)}</tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.run_id} className="border-t hover:bg-slate-50" data-testid="history-row">
              <td className="px-3 py-2"><Link className="text-blue-700 underline" href={`/runs/${r.run_id}`}>{r.request}</Link></td>
              <td className="px-3 py-2"><StatusBadge status={r.status} /></td><td className="px-3 py-2">{r.workflow ?? "-"}</td>
              <td className="px-3 py-2">{r.n_sources}</td><td className="px-3 py-2">{r.n_supported_claims}</td><td className="px-3 py-2">{r.n_conflicts}</td>
              <td className="px-3 py-2">{r.duration_s == null ? "-" : `${r.duration_s.toFixed(1)}s`}</td></tr>))}
            {rows.length === 0 && <tr><td colSpan={7} className="px-3 py-6 text-center text-slate-500">No runs match.</td></tr>}</tbody>
        </table>
      </CardBody></Card>
    </main>
  );
}
