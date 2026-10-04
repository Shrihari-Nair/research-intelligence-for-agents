"""/runs endpoints."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from sse_starlette.sse import EventSourceResponse

from research_os.api.schemas import CreateRun, RejectBody, ReportResponse, RunCreated, RunDetail, RunSummary, SourceContent
from research_os.api.service import RunRecord, RunService
from research_os.api.sse import event_stream
from research_os.artifacts import PDFUnavailable, ResearchReport, render_markdown, render_pdf

router = APIRouter(prefix="/runs", tags=["runs"])


def _service(request: Request) -> RunService:
    return request.app.state.service


def _get(request: Request, run_id: str) -> RunRecord:
    rec = _service(request).get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return rec


@router.post("", status_code=202, response_model=RunCreated)
def create_run(body: CreateRun, request: Request) -> RunCreated:
    """Start a research run in the background."""
    max_chars = request.app.state.max_text_chars
    text = body.text.strip()
    if not text or len(text) > max_chars:
        raise HTTPException(status_code=422, detail=f"text must be 1..{max_chars} characters")
    return RunCreated(run_id=_service(request).submit(text, body.budget_profile).run_id)


@router.get("", response_model=list[RunSummary])
def list_runs(request: Request) -> list[dict]:
    """Run history, newest first (live runs plus runs persisted before a restart)."""
    svc = _service(request)
    rows = [r.summary() for r in svc.list()]
    if svc.memory is not None:
        live = {r["run_id"] for r in rows}
        rows += [svc.persisted_summary(p) for p in svc.memory.list_runs() if p["run_id"] not in live]
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, request: Request) -> dict:
    """Status, plan, claims, budget usage."""
    rec = _service(request).get(run_id)
    if rec is not None:
        return rec.detail()
    persisted = _service(request).persisted_detail(run_id)
    if persisted is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return persisted


@router.post("/{run_id}/approve", status_code=202, response_model=RunCreated)
def approve_run(run_id: str, request: Request) -> RunCreated:
    """Approve a run that is AWAITING_APPROVAL (plan cost or publication). 409 if it is not waiting."""
    try:
        return RunCreated(run_id=_service(request).approve(run_id).run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found") from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post("/{run_id}/reject", status_code=202, response_model=RunCreated)
def reject_run(run_id: str, request: Request, body: RejectBody | None = None) -> RunCreated:
    """Reject a run that is AWAITING_APPROVAL; it ends CANCELLED."""
    try:
        return RunCreated(run_id=_service(request).reject(run_id, body.reason if body else None).run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found") from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/{run_id}/sources/{source_id}", response_model=SourceContent)
def get_source(run_id: str, source_id: str, request: Request) -> dict:
    """Full stored text of one source of a run (live or persisted)."""
    svc = _service(request)
    rec = svc.get(run_id)
    src = rec.ctx.sources.get(source_id) if rec is not None else None
    if src is None and rec is None and svc.memory is not None:
        src = next((s for s in svc.memory.get_sources(run_id) if s.source_id == source_id), None)
    if src is None:
        raise HTTPException(status_code=404, detail=f"source {source_id!r} not found in run {run_id!r}")
    return {"source_id": src.source_id, "title": src.title, "url": src.url, "source_type": src.source_type,
            "published_at": src.published_at.isoformat() if src.published_at else None, "content": src.content}


@router.get("/{run_id}/events")
async def run_events(run_id: str, request: Request, from_seq: int = Query(default=0, ge=0),
                     last_event_id: str | None = Header(default=None)) -> EventSourceResponse:
    """SSE stream of RunEvents. Resume with ?from_seq=N or the Last-Event-ID header."""
    rec = _get(request, run_id)
    start = from_seq
    if last_event_id is not None and last_event_id.isdigit():
        start = max(start, int(last_event_id) + 1)
    return EventSourceResponse(event_stream(rec, start))


def _report_dict(request: Request, run_id: str) -> tuple[str, dict]:
    svc = _service(request)
    rec = svc.get(run_id)
    if rec is None:
        row = svc.memory.get_run(run_id) if svc.memory is not None else None
        if row is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        if row["status"] in ("RUNNING", "PENDING"):
            raise HTTPException(status_code=409, detail="run is not finished yet")
        return row["status"], svc.memory.get_report(run_id) or {}
    if not rec.done:
        raise HTTPException(status_code=409, detail="run is not finished yet")
    return rec.status.value, rec.basic_report()


@router.get("/{run_id}/report")
def get_report(run_id: str, request: Request, format: Literal["json", "md", "pdf"] = Query(default="json")):
    """The ResearchReport as JSON (default), Markdown or PDF. 409 while the run is still going."""
    status, report = _report_dict(request, run_id)
    if format == "json":
        return ReportResponse(run_id=run_id, status=status, report=report)
    parsed = ResearchReport.model_validate(report)
    if format == "md":
        return Response(render_markdown(parsed), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="report-{run_id}.md"'})
    try:
        pdf = render_pdf(parsed)
    except PDFUnavailable as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="report-{run_id}.pdf"'})
