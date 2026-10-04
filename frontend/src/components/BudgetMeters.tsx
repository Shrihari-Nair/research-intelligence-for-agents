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
            <div className="flex justify-between text-xs text-slate-600"><span>{label}</span><span>{u}{l ? ` / ${l}` : ""}</span></div>
            <div className="h-1.5 rounded bg-slate-200"><div className={`h-1.5 rounded ${pct > 85 ? "bg-red-500" : pct > 60 ? "bg-amber-500" : "bg-blue-500"}`} style={{ width: `${pct}%` }} /></div>
          </div>
        );
      })}
      <div className="text-xs text-slate-500">elapsed {Number(budget.elapsed_seconds ?? 0).toFixed(1)}s{limits ? ` / ${limits.max_runtime_seconds}s` : ""}</div>
    </div>
  );
}
