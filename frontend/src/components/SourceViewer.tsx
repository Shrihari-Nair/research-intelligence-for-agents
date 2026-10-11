"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { getSource, type SourceContent } from "@/api/client";
import { findQuote, safeHref } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

/** Source text with the evidence quote highlighted. `quotes` are all quotes to mark; the first is scrolled into view. */
export function SourceViewer({ runId, sourceId, quotes }: { runId: string; sourceId: string; quotes: string[] }) {
  const [src, setSrc] = useState<SourceContent | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const mark = useRef<HTMLElement | null>(null);
  useEffect(() => { setSrc(null); setErr(null); getSource(runId, sourceId).then(setSrc).catch((e) => setErr(String(e))); }, [runId, sourceId]);
  const parts = useMemo(() => {
    if (!src) return [];
    const spans = quotes.map((q) => findQuote(src.content, q)).filter((s): s is [number, number] => !!s).sort((a, b) => a[0] - b[0]);
    const out: { text: string; hit: boolean }[] = [];
    let pos = 0;
    for (const [s, e] of spans) {
      if (s < pos) continue;
      out.push({ text: src.content.slice(pos, s), hit: false }, { text: src.content.slice(s, e), hit: true });
      pos = e;
    }
    out.push({ text: src.content.slice(pos), hit: false });
    return out;
  }, [src, quotes]);
  useEffect(() => { mark.current?.scrollIntoView({ block: "center" }); }, [parts]);
  if (err) return <div className="text-sm text-rose-300">{err}</div>;
  if (!src) return <div className="skeleton h-40 rounded-xl" aria-busy />;
  let firstHit = true;
  return (
    <div data-testid="source-viewer" className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <b>{src.title}</b><Badge>{src.source_type}</Badge>{src.published_at && <span className="font-mono text-xs text-fg-faint">{src.published_at.slice(0, 10)}</span>}
        <a className="font-mono text-xs text-volt underline-offset-4 hover:underline" href={safeHref(src.url)} target="_blank" rel="noreferrer">open original</a>
      </div>
      <div className="max-h-72 overflow-auto whitespace-pre-wrap rounded-xl border border-line bg-black/40 p-4 text-sm leading-6 text-fg-dim">
        {parts.map((p, i) => p.hit
          ? <mark key={i} ref={firstHit ? ((firstHit = false), (el: HTMLElement | null) => { mark.current = el; }) : undefined} data-testid="quote-highlight" className="rounded bg-volt px-0.5 text-ink shadow-[0_0_18px_rgba(198,255,61,.55)]">{p.text}</mark>
          : <span key={i}>{p.text}</span>)}
      </div>
    </div>
  );
}
