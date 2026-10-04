"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, TERMINAL, getRun, type RunDetail, type RunEvent } from "./client";

/** Live run state: detail from the REST API, events from SSE (EventSource reconnects with Last-Event-ID and the server resumes). */
export function useRun(runId: string) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDetail(await getRun(runId));
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [runId]);

  const refreshSoon = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(refresh, 150);
  }, [refresh]);

  useEffect(() => {
    setEvents([]);
    setDetail(null);
    refresh();
    const es = new EventSource(`${API_URL}/runs/${runId}/events`);
    es.addEventListener("run_event", (m) => {
      const ev = JSON.parse((m as MessageEvent).data) as RunEvent;
      setEvents((prev) => (prev.some((p) => p.seq === ev.seq) ? prev : [...prev, ev]));
      if (ev.type !== "started" || ev.node === "run") refreshSoon();
      if (ev.node === "run" && (ev.type === "finished" || ev.type === "failed")) es.close();
    });
    es.onerror = () => { /* the browser retries; polling below covers gaps */ };
    const poll = setInterval(() => { if (!detail || !TERMINAL.includes(detail.status)) refresh(); }, 2500);
    return () => { es.close(); clearInterval(poll); if (timer.current) clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  return { detail, events, error, refresh, done: !!detail && TERMINAL.includes(detail.status) };
}
