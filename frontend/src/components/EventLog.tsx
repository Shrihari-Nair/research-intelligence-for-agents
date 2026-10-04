import type { RunEvent } from "@/api/client";

const COLOR: Record<string, string> = { failed: "text-red-700", recovery: "text-amber-700", retry: "text-amber-700", budget: "text-amber-700", approval: "text-violet-700", finished: "text-green-700" };

export function EventLog({ events }: { events: RunEvent[] }) {
  return (
    <pre className="h-72 overflow-auto rounded border bg-slate-50 p-2 text-[11px] leading-4" data-testid="event-log">
      {events.map((e) => (
        <div key={e.seq} className={COLOR[e.type] ?? "text-slate-700"}>
          {`${String(e.seq).padStart(3)}  ${e.type.padEnd(8)} ${e.node.padEnd(22)} ${JSON.stringify(e.payload).slice(0, 120)}`}
        </div>
      ))}
    </pre>
  );
}
