import { cn } from "@/lib/utils";

const TONES: Record<string, string> = {
  green: "bg-emerald-400/10 text-emerald-300 border-emerald-400/25",
  red: "bg-rose-500/10 text-rose-300 border-rose-400/25",
  amber: "bg-amber-400/10 text-amber-300 border-amber-400/25",
  blue: "bg-volt/10 text-volt border-volt/30",
  slate: "bg-white/5 text-fg-dim border-white/10",
  violet: "bg-fuchsia-400/10 text-fuchsia-300 border-fuchsia-400/25",
};

export function Badge({ tone = "slate", className, children }: { tone?: keyof typeof TONES | string; className?: string; children: React.ReactNode }) {
  return <span className={cn("inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[11px] font-medium tracking-wide", TONES[tone] ?? TONES.slate, className)}>{children}</span>;
}
