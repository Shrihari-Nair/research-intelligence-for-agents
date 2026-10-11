"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { createRun } from "@/api/client";
import { Button } from "@/components/ui/button";

const EXAMPLES = [
  "Compare LangGraph and CrewAI for multi-agent workflows",
  "What does the research say about speculative decoding?",
  "How big is the vector database market and how fast is it growing?",
];

const PROMISES = ["every claim carries a verbatim quote", "hard budgets on searches and LLM calls", "a human approves anything expensive", "conflicting sources are shown, never merged", "unsupported claims are dropped before you see them"];

export default function Workspace() {
  const router = useRouter();
  const [text, setText] = useState("");
  const [profile, setProfile] = useState<"" | "small" | "medium" | "large">("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function start() {
    setBusy(true); setError(null);
    try { router.push(`/runs/${await createRun({ text, budget_profile: profile || null })}`); }
    catch (e) { setError(String(e)); setBusy(false); }
  }
  return (
    <main className="relative">
      <section className="pt-10 sm:pt-16">
        <div className="animate-fade-up font-mono text-xs uppercase tracking-[0.25em] text-volt">
          <span className="relative mr-3 inline-flex h-2 w-2 align-middle"><span className="absolute inline-flex h-full w-full animate-ping2 rounded-full bg-volt" /><span className="relative inline-flex h-2 w-2 rounded-full bg-volt" /></span>
          agents online
        </div>
        <h1 className="mt-5 max-w-5xl animate-fade-up text-[clamp(2.8rem,8.5vw,7rem)] font-extrabold leading-[0.92] tracking-[-0.045em]" style={{ animationDelay: "80ms" }}>
          What do you want <br />
          <span className="text-volt-grad">researched</span><span className="text-volt">?</span>
        </h1>
        <p className="mt-6 max-w-xl animate-fade-up text-lg leading-relaxed text-fg-dim" style={{ animationDelay: "160ms" }}>
          Ask anything. A router picks the workflow, agents search and read, and a verifier checks each claim against its source quote.
        </p>
      </section>

      <section className="mt-12 max-w-3xl animate-fade-up" style={{ animationDelay: "240ms" }}>
        <div className="glow-frame shadow-glow">
          <div className="bg-ink-2 p-5 sm:p-6">
            <label htmlFor="req" className="mb-2 flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.18em] text-fg-faint"><span>Type your research question</span><span className="hidden sm:inline normal-case tracking-normal">⌘/Ctrl + Enter to start</span></label>
            <textarea id="req" data-testid="request-input" rows={3} value={text} autoFocus
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && text.trim() && !busy) start(); }}
              className="w-full resize-none rounded-2xl border border-white/15 bg-black/40 px-4 py-3.5 text-lg leading-snug tracking-tight text-fg caret-volt shadow-[inset_0_2px_12px_rgba(0,0,0,.5)] transition-colors placeholder:text-fg-faint hover:border-volt/40 focus:border-volt focus:outline-none focus:ring-2 focus:ring-volt/30 sm:text-xl" placeholder="e.g. Compare Redis and Memcached for session caching…" />
            <div className="mt-4 font-mono text-[11px] uppercase tracking-[0.18em] text-fg-faint">…or start from an example</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {EXAMPLES.map((x) => (
                <button key={x} onClick={() => setText(x)} className={`rounded-lg border px-3 py-1.5 text-left text-xs transition-all duration-200 hover:-translate-y-0.5 hover:border-volt/60 hover:text-volt ${x === text ? "border-volt/60 bg-volt/10 text-volt" : "border-white/10 text-fg-dim"}`}>{x}</button>
              ))}
            </div>
            <div className="mt-6 flex flex-wrap items-center gap-4 border-t border-line pt-5">
              <label className="flex items-center gap-3 text-sm text-fg-dim">Budget
                <select data-testid="budget-select" value={profile} onChange={(e) => setProfile(e.target.value as typeof profile)}
                  className="rounded-lg border border-white/10 bg-ink-3 px-3 py-1.5 font-mono text-xs text-fg transition-colors hover:border-volt/50 focus:border-volt focus:outline-none">
                  <option value="">workflow default</option><option value="small">small</option><option value="medium">medium</option><option value="large">large</option>
                </select>
              </label>
              <Button data-testid="start-button" className="ml-auto" disabled={busy || !text.trim()} onClick={start}>
                {busy ? "Starting…" : <>Start <span aria-hidden>→</span></>}
              </Button>
            </div>
            {error && <div role="alert" className="mt-3 rounded-lg border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">{error}</div>}
          </div>
        </div>
      </section>

      <div className="relative mt-20 overflow-hidden border-y border-line py-4 [mask-image:linear-gradient(90deg,transparent,#000_12%,#000_88%,transparent)]" aria-hidden>
        <div className="flex w-max animate-ticker gap-12 font-mono text-sm uppercase tracking-[0.18em] text-fg-faint">
          {[...PROMISES, ...PROMISES].map((p, i) => <span key={i} className="flex items-center gap-12 whitespace-nowrap">{p}<span className="text-volt">✦</span></span>)}
        </div>
      </div>
    </main>
  );
}
