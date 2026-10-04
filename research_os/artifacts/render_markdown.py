"""Markdown renderer (Jinja2). Refuses reports that violate citation integrity."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from research_os.artifacts.report import ResearchReport, check_integrity

_ENV = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")), undefined=StrictUndefined,
                   keep_trailing_newline=True, trim_blocks=False, autoescape=False)


def render_markdown(report: ResearchReport) -> str:
    """Report -> Markdown with numbered citations [n] that map to the Evidence section."""
    check_integrity(report)
    num = {e.claim_id: i for i, e in enumerate(report.evidence, 1)}
    srcs = {s.source_id: s for s in report.sources}
    return _ENV.get_template("report.md.j2").render(r=report, num=num, srcs=srcs)
