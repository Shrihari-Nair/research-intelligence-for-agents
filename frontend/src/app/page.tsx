"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { createRun } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";

const EXAMPLES = [
  "Compare LangGraph and CrewAI for multi-agent workflows",
  "What does the research say about speculative decoding?",
  "How big is the vector database market and how fast is it growing?",
];

export default function Workspace() {
  const router = useRouter();
  const [text, setText] = useState(EXAMPLES[0]);
  const [profile, setProfile] = useState<"" | "small" | "medium" | "large">("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function start() {
    setBusy(true); setError(null);
    try { router.push(`/runs/${await createRun({ text, budget_profile: profile || null })}`); }
    catch (e) { setError(String(e)); setBusy(false); }
  }
  return (
    <main className="mx-auto max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">What do you want to research?</h1>
      <Card>
        <CardHeader>Research request</CardHeader>
        <CardBody className="space-y-3">
          <textarea data-testid="request-input" className="w-full rounded border p-2 text-sm" rows={4} value={text} onChange={(e) => setText(e.target.value)} />
          <div className="flex flex-wrap gap-2">{EXAMPLES.map((x) => <button key={x} className="rounded-full border px-3 py-1 text-xs text-slate-600 hover:bg-slate-100" onClick={() => setText(x)}>{x}</button>)}</div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-slate-600">Budget
              <select data-testid="budget-select" className="ml-2 rounded border p-1 text-sm" value={profile} onChange={(e) => setProfile(e.target.value as typeof profile)}>
                <option value="">workflow default</option><option value="small">small</option><option value="medium">medium</option><option value="large">large</option>
              </select></label>
            <Button data-testid="start-button" disabled={busy || !text.trim()} onClick={start}>{busy ? "Starting…" : "Start"}</Button>
            {error && <span className="text-sm text-red-700">{error}</span>}
          </div>
        </CardBody>
      </Card>
    </main>
  );
}
