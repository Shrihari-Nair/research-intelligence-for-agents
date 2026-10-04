import { cn } from "@/lib/utils";

const TONES: Record<string, string> = {
  green: "bg-green-100 text-green-800 border-green-200",
  red: "bg-red-100 text-red-800 border-red-200",
  amber: "bg-amber-100 text-amber-800 border-amber-200",
  blue: "bg-blue-100 text-blue-800 border-blue-200",
  slate: "bg-slate-100 text-slate-700 border-slate-200",
  violet: "bg-violet-100 text-violet-800 border-violet-200",
};

export function Badge({ tone = "slate", className, children }: { tone?: keyof typeof TONES | string; className?: string; children: React.ReactNode }) {
  return <span className={cn("inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium", TONES[tone] ?? TONES.slate, className)}>{children}</span>;
}
