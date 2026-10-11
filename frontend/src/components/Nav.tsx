"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "New research", test: undefined },
  { href: "/history", label: "History", test: "nav-history" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-4 z-30 px-4">
      <nav className="mx-auto flex max-w-5xl items-center gap-2 rounded-2xl border border-line bg-ink-2/70 py-2 pl-4 pr-2 shadow-panel backdrop-blur-xl">
        <Link href="/" className="mr-auto flex items-center gap-2.5 font-semibold tracking-tight">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden><rect width="32" height="32" rx="9" fill="#c6ff3d" /><path d="M8 22V10h8.5a4 4 0 0 1 0 8H8m8.5 0L24 22" fill="none" stroke="#070a09" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg>
          <span className="whitespace-nowrap"><span className="hidden sm:inline">Research Intelligence </span><span className="text-volt">OS</span></span>
        </Link>
        {LINKS.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link key={l.href} href={l.href} data-testid={l.test} aria-current={active ? "page" : undefined}
              className={cn("whitespace-nowrap rounded-xl px-3.5 py-1.5 text-sm font-medium transition-all duration-200", active ? "bg-white/10 text-fg" : "text-fg-dim hover:bg-white/5 hover:text-fg")}>
              {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
