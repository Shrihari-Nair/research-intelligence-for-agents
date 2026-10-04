"""Wire a ResearchStore into a RunContext: incremental persistence of sources and claims, plus final run record."""
from __future__ import annotations

from research_os.artifacts.report import build_report


def attach_memory(ctx, store, request: str, llm_backend: str = "unknown") -> None:
    """Create the run row and persist sources/claims as they appear (a crash keeps everything gathered so far)."""
    ctx.config.setdefault("llm_backend", llm_backend)
    store.create_run(ctx.run_id, request, llm_backend=llm_backend)
    ctx.memory = store
    ctx.sources.on_add = lambda src: store.add_sources(ctx.run_id, [src])
    for src in ctx.sources.all():  # sources gathered before the hook was attached
        store.add_sources(ctx.run_id, [src])

    def on_step(step, outputs: dict) -> None:
        claims = outputs.get("claims") if isinstance(outputs, dict) else None
        if claims:
            store.add_claims(ctx.run_id, claims)

    ctx.step_listeners.append(on_step)


def finalize_run(ctx, store, state: dict, request: str | None = None, status: str | None = None, error: str | None = None) -> dict:
    """Persist the terminal status, budget, router output and report. Returns the report dict."""
    request = request or state.get("request", "")
    final_status = status or state.get("status", "FAILED")
    report = build_report(request, state, ctx, status=final_status, error=error).model_dump(mode="json")
    ro, dec = state.get("router_output"), state.get("decision")
    store.finish_run(ctx.run_id, final_status, workflow=dec.workflow if dec else None, budget=ctx.budget.snapshot(),
                     router=ro.model_dump(mode="json") if ro else None, report=report)
    return report
