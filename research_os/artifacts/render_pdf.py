"""PDF renderer behind an interface. Default: WeasyPrint (HTML+CSS). See docs/PDF_SPIKE.md."""
from __future__ import annotations

from typing import Protocol

from research_os.artifacts.render_markdown import render_markdown
from research_os.artifacts.report import ResearchReport

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'DejaVu Sans', 'Noto Sans', sans-serif; font-size: 10pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin-bottom: 0.2em; } h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 1.4em; }
h3 { font-size: 11pt; } code { font-size: 9pt; background: #f3f3f3; padding: 0 2px; } a { color: #1a4fa0; text-decoration: none; }
"""


class PDFUnavailable(RuntimeError):
    """The PDF backend cannot run on this machine."""


class PDFRenderer(Protocol):
    """Anything that turns a report into PDF bytes."""

    def render(self, report: ResearchReport) -> bytes: ...


class WeasyPrintRenderer:
    """Markdown -> HTML -> PDF with WeasyPrint."""

    def render(self, report: ResearchReport) -> bytes:
        md = render_markdown(report)  # also enforces citation integrity
        try:
            import markdown
            from weasyprint import CSS as WCSS
            from weasyprint import HTML
        except (ImportError, OSError) as e:
            raise PDFUnavailable(f"WeasyPrint is not usable here: {e}") from e
        html = markdown.markdown(md, extensions=["extra", "sane_lists"])
        return HTML(string=f"<html><body>{html}</body></html>").write_pdf(stylesheets=[WCSS(string=CSS)])


def render_pdf(report: ResearchReport, renderer: PDFRenderer | None = None) -> bytes:
    """Report -> PDF bytes (WeasyPrint unless another renderer is given)."""
    return (renderer or WeasyPrintRenderer()).render(report)
