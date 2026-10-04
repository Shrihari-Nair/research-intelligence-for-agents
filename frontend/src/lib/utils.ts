import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Whitespace-tolerant, case-sensitive search for a verbatim quote; returns [start, end] in the ORIGINAL text or null. */
export function findQuote(content: string, quote: string): [number, number] | null {
  const tokens = quote.replace(/ /g, " ").trim().split(/\s+/).filter(Boolean);
  if (!tokens.length) return null;
  const esc = tokens.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const m = new RegExp(esc.join("\\s+")).exec(content.replace(/ /g, " "));
  return m ? [m.index, m.index + m[0].length] : null;
}

/** Only http(s) URLs may become links; anything else (e.g. javascript:) is dropped. */
export function safeHref(url?: string): string | undefined {
  try {
    const u = new URL(url ?? "");
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : undefined;
  } catch {
    return undefined;
  }
}
