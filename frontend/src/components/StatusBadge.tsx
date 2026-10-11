import { Badge } from "@/components/ui/badge";

const TONE: Record<string, string> = {
  SUCCESS: "green", PARTIAL: "amber", FAILED: "red", CANCELLED: "slate", RUNNING: "blue", PENDING: "slate", AWAITING_APPROVAL: "violet",
  succeeded: "green", failed: "red", skipped: "amber", running: "blue", pending: "slate",
  SUPPORTED: "green", CONTRADICTED: "red", UNSUPPORTED: "amber", UNVERIFIED: "slate",
};

export function StatusBadge({ status }: { status: string }) {
  const tone = TONE[status] ?? "slate";
  const live = status === "RUNNING" || status === "running";
  return <Badge tone={tone}>{live && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}{status.replace("_", " ")}</Badge>;
}
