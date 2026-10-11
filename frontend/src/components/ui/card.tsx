"use client";
import { cn } from "@/lib/utils";

/** Glass panel; the border lights up under the cursor via --mx/--my. */
export function Card({ className, onMouseMove, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("panel rounded-2xl", className)}
      onMouseMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        e.currentTarget.style.setProperty("--mx", `${e.clientX - r.left}px`);
        e.currentTarget.style.setProperty("--my", `${e.clientY - r.top}px`);
        onMouseMove?.(e);
      }}
      {...props}
    />
  );
}
export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex items-center gap-2 border-b border-line px-5 py-3 font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-fg-dim before:h-1.5 before:w-1.5 before:rounded-full before:bg-volt before:shadow-[0_0_10px_#c6ff3d]", className)} {...props} />;
}
export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}
