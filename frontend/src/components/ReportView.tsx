"use client";
import { safeHref } from "@/lib/utils";
import { useEffect, useMemo, useState } from "react";
import { getReport, reportUrl, type Report, type RunDetail } from "@/api/client";
import { SourceViewer } from "@/components/SourceViewer";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

export function ReportView({ detail }: { detail: RunDetail }) {
  const [report, setReport] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [cite, setCite] = useState<string | null>(null);
  useEffect(() => { getReport(detail.run_id).then(setReport).catch((e) => setErr(String(e))); }, [detail.run_id, detail.status]);
  const num = useMemo(() => new Map((report?.evidence ?? []).map((e, i) => [e.claim_id, i + 1])), [report]);
  const srcById = useMemo(() => new Map((report?.sources ?? []).map((s) => [s.source_id, s])), [report]);
  if (err) return <div className="text-sm text-rose-300">{err}</div>;
  if (!report) return <div className="space-y-4" aria-busy><div className="skeleton h-32 rounded-2xl" /><div className="skeleton h-48 rounded-2xl" /></div>;
  const ev = report.evidence.find((e) => e.claim_id === cite);
  const downloadJson = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `report-${detail.run_id}.json`; a.click(); URL.revokeObjectURL(url);
  };
  const Cites = ({ ids }: { ids: string[] }) => (
    <>{ids.map((id) => <button key={id} data-testid="citation" className="ml-1 rounded-md bg-volt/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-volt transition-all hover:-translate-y-0.5 hover:bg-volt hover:text-ink" onClick={() => setCite(id)}>[{num.get(id)}]</button>)}</>
  );
  return (
    <div className="space-y-5" data-testid="report">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={report.status} />
        <a data-testid="download-md" href={reportUrl(detail.run_id, "md")} download><Button variant="outline" size="sm">Download Markdown</Button></a>
        <a data-testid="download-pdf" href={reportUrl(detail.run_id, "pdf")} download><Button variant="outline" size="sm">Download PDF</Button></a>
        <Button data-testid="download-json" variant="outline" size="sm" onClick={downloadJson}>Download JSON</Button>
      </div>
      <Card><CardHeader>Executive summary</CardHeader><CardBody className="max-w-3xl text-lg leading-8 text-fg">{report.executive_summary}</CardBody></Card>
      <Card>
        <CardHeader>Findings</CardHeader>
        <CardBody>
          {report.findings.length === 0 ? <div className="text-sm text-fg-dim">No verified findings.</div> : (
            <ul className="list-disc space-y-2.5 pl-5 text-[15px] leading-7 marker:text-volt">{report.findings.map((f, i) => <li key={i}>{f.statement}<Cites ids={f.claim_ids} /></li>)}</ul>
          )}
        </CardBody>
      </Card>
      {report.comparison && (
        <Card><CardHeader>Comparison</CardHeader><CardBody className="grid gap-4 md:grid-cols-2">
          {report.comparison.map((row) => (
            <div key={row.entity}><div className="mb-2 text-lg font-bold text-volt">{row.entity}</div>
              <ul className="list-disc space-y-1.5 pl-5 text-sm leading-6 marker:text-volt">{row.claims.map((c) => <li key={c.claim_id}>{c.statement}<Cites ids={[c.claim_id]} /></li>)}</ul></div>
          ))}
        </CardBody></Card>
      )}
      {report.contradictions.length > 0 && (
        <Card><CardHeader>Contradictions</CardHeader><CardBody className="space-y-3 text-sm">
          <div className="text-fg-dim">Sources disagree. Both sides are shown; neither is presented as a finding.</div>
          {report.contradictions.map((k) => (
            <div key={k.subject} className="grid gap-2 md:grid-cols-2">{k.sides.map((s) => (
              <div key={s.stance} className="rounded-xl border border-rose-400/30 bg-rose-500/[0.07] p-3"><div className="mb-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-rose-300">{s.stance}</div>
                {s.claims.map((c) => <div key={c.claim_id}>{c.claim}<div className="text-xs italic text-fg-faint">“{c.evidence_quote}”</div></div>)}</div>))}</div>
          ))}
        </CardBody></Card>
      )}
      <Card><CardHeader>Evidence</CardHeader><CardBody>
        <ol className="space-y-3 text-sm leading-6 text-fg-dim [&_b]:text-fg">{report.evidence.map((e, i) => (
          <li key={e.claim_id} id={`ev-${i + 1}`}>[{i + 1}] <b>{e.claim}</b> — <i>“{e.evidence_quote}”</i> — <a className="text-volt underline-offset-4 hover:underline" href={safeHref(srcById.get(e.source_id)?.url)} target="_blank" rel="noreferrer">{srcById.get(e.source_id)?.title}</a></li>))}
        </ol>
        {report.evidence.length === 0 && <div className="text-sm text-fg-dim">No verified evidence.</div>}
      </CardBody></Card>
      <Card><CardHeader>Limitations</CardHeader><CardBody><ul className="list-disc space-y-1.5 pl-5 text-sm leading-6 marker:text-volt">{report.limitations.map((l) => <li key={l}>{l}</li>)}</ul></CardBody></Card>
      <Dialog open={!!ev} onOpenChange={(o) => !o && setCite(null)}>
        <DialogContent data-testid="citation-dialog">
          <DialogTitle className="text-xl font-bold tracking-tight">Citation [{ev ? num.get(ev.claim_id) : ""}]</DialogTitle>
          <DialogDescription className="mt-2 text-sm text-fg-dim">{ev?.claim}</DialogDescription>
          {ev && <div className="mt-3"><SourceViewer runId={detail.run_id} sourceId={ev.source_id} quotes={[ev.evidence_quote]} /></div>}
          <div className="mt-3 text-right"><Button variant="outline" size="sm" onClick={() => setCite(null)}>Close</Button></div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
