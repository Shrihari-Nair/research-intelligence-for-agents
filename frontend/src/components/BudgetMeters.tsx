import type { RunDetail } from "@/api/client";

const ROWS: [string, string, string][] = [
  ["steps", "max_steps", "Steps"], ["tool_calls", "max_tool_calls", "Tool calls"], ["searches", "max_searches", "Searches"], ["llm_calls", "max_llm_calls", "LLM calls"],
];

export function BudgetMeters({ budget }: { budget: RunDetail["budget"] }) {
  const limits = (budget.limits ?? null) as Record<string, number> | null;
  return (
    <div className="space-y-2" data-testid="budget-meters">
      {ROWS.map(([used, lim, label]) => {
        const u = Number(budget[used] ?? 0);
        const l = limits?.[lim];
        const pct = l ? Math.min(100, (u / l) * 100) : 0;
        return (
          <div key={used}>
            <div className="mb-1 flex justify-between font-mono text-[11px] uppercase tracking-wider text-fg-dim"><span>{label}</span><span className="tnum text-fg">{u}{l ? ` / ${l}` : ""}</span></div>
            <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]"><div className={`h-full rounded-full transition-[width] duration-700 ${pct > 85 ? "bg-rose-400 shadow-[0_0_12px_#fb7185]" : pct > 60 ? "bg-amber-300 shadow-[0_0_12px_#fcd34d]" : "bg-volt shadow-[0_0_12px_#c6ff3d]"}`} style={{ width: `${pct}%` }} /></div>
          </div>
        );
      })}
      <div className="tnum pt-1 font-mono text-[11px] text-fg-faint">elapsed {Number(budget.elapsed_seconds ?? 0).toFixed(1)}s{limits ? ` / ${limits.max_runtime_seconds}s` : ""}</div>
    </div>
  );
}
