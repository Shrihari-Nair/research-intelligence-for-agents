"use client";
import { useEffect, useState } from "react";
import { approveRun, rejectRun, type RunDetail } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

export function ApprovalDialog({ detail, onResolved }: { detail: RunDetail; onResolved: () => void }) {
  const a = detail.approval as Record<string, unknown> | null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [left, setLeft] = useState<number | null>(null);
  useEffect(() => {
    if (!a?.expires_at) return;
    const tick = () => setLeft(Math.max(0, Math.round((new Date(String(a.expires_at)).getTime() - Date.now()) / 1000)));
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, [a?.expires_at]);
  if (detail.status !== "AWAITING_APPROVAL" || !a) return null;
  const budget = (a.budget ?? null) as Record<string, number> | null;
  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true); setErr(null);
    try { await fn(); onResolved(); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const publish = a.kind === "publish";
  return (
    <Dialog open>
      <DialogContent onInteractOutside={(e) => e.preventDefault()} onEscapeKeyDown={(e) => e.preventDefault()} data-testid="approval-dialog">
        <DialogTitle className="text-lg font-semibold">{publish ? "Approve publishing this report?" : "Approval required before this run starts"}</DialogTitle>
        <DialogDescription className="mt-1 text-sm text-slate-600">
          {publish ? "The research is finished. Nothing is released until you approve." : "This plan is expensive. No search or LLM call happens until you approve."}
        </DialogDescription>
        <ul className="mt-3 list-disc space-y-1 pl-5 text-sm">{((a.reasons as string[]) ?? []).map((r) => <li key={r}>{r}</li>)}</ul>
        {!publish && (
          <div className="mt-3 grid grid-cols-2 gap-2 rounded bg-slate-50 p-3 text-sm">
            <div>Searches: <b>{String(a.searches)}</b></div><div>Expected sources: <b>{String(a.expected_sources)}</b></div>
            <div>LLM calls (est.): <b>{String(a.llm_calls)}</b></div><div>Plan steps: <b>{String(a.steps)}</b></div>
            {budget && <div className="col-span-2 text-xs text-slate-600">Budget that will apply: {budget.max_steps} steps · {budget.max_tool_calls} tool calls · {budget.max_searches} searches · {budget.max_llm_calls} LLM calls · {budget.max_runtime_seconds}s</div>}
          </div>
        )}
        {left !== null && <div className="mt-2 text-xs text-slate-500">Expires in {left}s</div>}
        {err && <div className="mt-2 text-sm text-red-700">{err}</div>}
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="outline" disabled={busy} onClick={() => act(() => rejectRun(detail.run_id, "cancelled by user"))}>Cancel run</Button>
          <Button disabled={busy} onClick={() => act(() => approveRun(detail.run_id))}>Approve</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
