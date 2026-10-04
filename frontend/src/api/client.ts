import type { components } from "./schema";

export type RunDetail = components["schemas"]["RunDetail"];
export type RunSummary = components["schemas"]["RunSummary"];
export type StepView = components["schemas"]["StepView"];
export type CreateRun = components["schemas"]["CreateRun"];
export type SourceContent = components["schemas"]["SourceContent"];

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface RunEvent {
  run_id: string;
  seq: number;
  ts: string;
  node: string;
  type: "started" | "finished" | "failed" | "retry" | "recovery" | "budget" | "approval" | "info";
  payload: Record<string, unknown>;
}

/** The canonical ResearchReport (research_os/artifacts/report.py). */
export interface Report {
  question: string;
  status: "SUCCESS" | "PARTIAL" | "FAILED" | "CANCELLED";
  executive_summary: string;
  methodology: Record<string, unknown>;
  findings: { statement: string; claim_ids: string[] }[];
  comparison: { entity: string; claims: { claim_id: string; statement: string }[] }[] | null;
  evidence: { claim_id: string; claim: string; source_id: string; evidence_quote: string; verdict: string }[];
  contradictions: Conflict[];
  limitations: string[];
  sources: { source_id: string; title: string; url: string; source_type: string; published_at: string | null }[];
  run_metadata: Record<string, unknown>;
}

export interface ClaimView {
  claim_id: string;
  claim: string;
  source_id: string;
  evidence_quote: string;
  verdict: "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED" | "UNVERIFIED";
  verdict_rationale?: string | null;
}

export interface Conflict {
  subject: string;
  sides: { stance: string; claims: { claim_id: string; claim: string; source_id: string; source_url: string; evidence_quote: string }[] }[];
}

async function j<T>(r: Response, what: string): Promise<T> {
  if (!r.ok) throw new Error(`${what} failed: ${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

export const createRun = async (body: CreateRun) =>
  (await j<{ run_id: string }>(await fetch(`${API_URL}/runs`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }), "POST /runs")).run_id;
export const getRun = async (id: string) => j<RunDetail>(await fetch(`${API_URL}/runs/${id}`), `GET /runs/${id}`);
export const listRuns = async () => j<RunSummary[]>(await fetch(`${API_URL}/runs`), "GET /runs");
export const getReport = async (id: string) => (await j<{ report: Report }>(await fetch(`${API_URL}/runs/${id}/report`), "GET report")).report;
export const getSource = async (id: string, sourceId: string) => j<SourceContent>(await fetch(`${API_URL}/runs/${id}/sources/${sourceId}`), "GET source");
export const approveRun = async (id: string) => j(await fetch(`${API_URL}/runs/${id}/approve`, { method: "POST" }), "approve");
export const rejectRun = async (id: string, reason?: string) =>
  j(await fetch(`${API_URL}/runs/${id}/reject`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ reason: reason ?? null }) }), "reject");
export const reportUrl = (id: string, format: "json" | "md" | "pdf") => `${API_URL}/runs/${id}/report?format=${format}`;

export const TERMINAL = ["SUCCESS", "PARTIAL", "FAILED", "CANCELLED"];
