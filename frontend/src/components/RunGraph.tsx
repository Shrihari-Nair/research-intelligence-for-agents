"use client";
import { useMemo } from "react";
import { Background, Controls, Handle, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { RunDetail, RunEvent } from "@/api/client";

type Status = "pending" | "running" | "succeeded" | "failed" | "skipped";
const FILL: Record<Status, string> = {
  pending: "#e2e8f0", running: "#bfdbfe", succeeded: "#bbf7d0", failed: "#fecaca", skipped: "#fde68a",
};

function StageNode({ data }: NodeProps) {
  const d = data as { label: string; sub?: string; status: Status; kind?: string };
  return (
    <div
      data-testid={`node-${d.label}`}
      data-status={d.status}
      className={`rounded-md border px-3 py-1.5 text-xs shadow-sm ${d.kind === "recovery" ? "border-amber-400" : "border-slate-400"}`}
      style={{ background: FILL[d.status], minWidth: 120 }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="font-semibold">{d.label}</div>
      {d.sub && <div className="text-[10px] text-slate-600">{d.sub}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { stage: StageNode };

function stageStatus(events: RunEvent[], node: string): Status {
  let s: Status = "pending";
  for (const e of events) {
    if (e.node !== node) continue;
    if (e.type === "started") s = "running";
    else if (e.type === "finished") s = "succeeded";
    else if (e.type === "failed") s = "failed";
  }
  return s;
}

export function buildGraph(detail: RunDetail, events: RunEvent[]): { nodes: Node[]; edges: Edge[] } {
  const X = 230, Y = 78;
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const add = (id: string, label: string, col: number, row: number, status: Status, sub?: string, kind?: string) =>
    nodes.push({ id, type: "stage", position: { x: col * X, y: row * Y }, data: { label, sub, status, kind } });
  const link = (a: string, b: string, animated = false) => edges.push({ id: `${a}->${b}`, source: a, target: b, animated });

  add("router", "router", 0, 0, stageStatus(events, "router"), detail.router ? `${detail.router.intent} ${(Number(detail.router.confidence) * 100).toFixed(0)}%` : undefined);
  add("supervisor", "supervisor", 1, 0, stageStatus(events, "supervisor"), detail.decision ? String(detail.decision.workflow) : undefined);
  add("planner", "planner", 2, 0, stageStatus(events, "planner"));
  link("router", "supervisor"); link("supervisor", "planner");
  let first = "planner";
  let colBase = 3;
  if (events.some((e) => e.type === "approval")) {
    const waiting = detail.status === "AWAITING_APPROVAL";
    add("approval", "approval", 3, 0, waiting ? "running" : events.some((e) => e.type === "approval" && e.payload.event === "rejected" || e.payload.event === "expired") ? "failed" : "succeeded", waiting ? "waiting for you" : undefined);
    link("planner", "approval"); first = "approval"; colBase = 4;
  }
  const steps = detail.plan;
  const depth = new Map<string, number>();
  const byId = new Map(steps.map((s) => [s.id, s]));
  const d = (id: string): number => {
    if (depth.has(id)) return depth.get(id)!;
    const s = byId.get(id);
    const deps = s?.depends_on ?? [];
    const v = deps.length === 0 ? 0 : 1 + Math.max(...deps.map(d));
    depth.set(id, v);
    return v;
  };
  const rowsAt = new Map<number, number>();
  for (const s of steps) {
    const col = d(s.id);
    const row = rowsAt.get(col) ?? 0;
    rowsAt.set(col, row + 1);
    add(`step:${s.id}`, s.id, colBase + col, row - 0, s.status as Status, s.agent);
    const deps = s.depends_on ?? [];
    if (deps.length === 0) link(first, `step:${s.id}`, s.status === "running");
    for (const dep of deps) link(`step:${dep}`, `step:${s.id}`, s.status === "running");
  }
  const maxCol = steps.length ? Math.max(...steps.map((s) => d(s.id))) : -1;
  const finalStatus: Status = detail.status === "SUCCESS" ? "succeeded" : detail.status === "FAILED" ? "failed" : detail.status === "PARTIAL" ? "skipped" : detail.status === "CANCELLED" ? "failed" : "pending";
  add("report", "report", colBase + maxCol + 1, 0, finalStatus, detail.status);
  const lastIds = steps.filter((s) => !steps.some((o) => (o.depends_on ?? []).includes(s.id)));
  for (const s of lastIds) link(`step:${s.id}`, "report");
  if (!steps.length) link(first, "report");
  // recovery decisions hang below the step they concern
  events.filter((e) => e.type === "recovery").forEach((e, i) => {
    const step = String(e.payload.step ?? "");
    const target = nodes.find((n) => n.id === `step:${step}`);
    if (!target) return;
    const id = `recovery-${e.seq}`;
    nodes.push({ id, type: "stage", position: { x: target.position.x + 20, y: target.position.y + 56 + (i % 3) * 8 }, data: { label: `↻ ${String(e.payload.action)}`, status: "skipped", kind: "recovery" } });
    edges.push({ id: `${id}-e`, source: `step:${step}`, target: id, style: { strokeDasharray: "4 3" } });
  });
  return { nodes, edges };
}

export function RunGraph({ detail, events }: { detail: RunDetail; events: RunEvent[] }) {
  const { nodes, edges } = useMemo(() => buildGraph(detail, events), [detail, events]);
  return (
    <div className="h-[460px] rounded-lg border bg-white" data-testid="run-graph">
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.15 }} nodesDraggable={false} proOptions={{ hideAttribution: true }}>
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
