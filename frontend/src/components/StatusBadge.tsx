import { Badge } from "@/components/ui/badge";

const TONE: Record<string, string> = {
  SUCCESS: "green", PARTIAL: "amber", FAILED: "red", CANCELLED: "slate", RUNNING: "blue", PENDING: "slate", AWAITING_APPROVAL: "violet",
  succeeded: "green", failed: "red", skipped: "amber", running: "blue", pending: "slate",
  SUPPORTED: "green", CONTRADICTED: "red", UNSUPPORTED: "amber", UNVERIFIED: "slate",
};

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={TONE[status] ?? "slate"}>{status.replace("_", " ")}</Badge>;
}
