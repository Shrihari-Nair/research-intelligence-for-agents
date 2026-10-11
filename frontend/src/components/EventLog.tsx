import type { RunEvent } from "@/api/client";

const COLOR: Record<string, string> = { failed: "text-rose-400", recovery: "text-amber-300", retry: "text-amber-300", budget: "text-amber-300", approval: "text-fuchsia-300", finished: "text-volt" };

export function EventLog({ events }: { events: RunEvent[] }) {
  return (
    <pre className="h-72 overflow-auto rounded-xl border border-line bg-black/40 p-3 font-mono text-[11px] leading-5" data-testid="event-log">
      {events.map((e) => (
        <div key={e.seq} className={COLOR[e.type] ?? "text-fg-dim"}>
          {`${String(e.seq).padStart(3)}  ${e.type.padEnd(8)} ${e.node.padEnd(22)} ${JSON.stringify(e.payload).slice(0, 120)}`}
        </div>
      ))}
    </pre>
  );
}
