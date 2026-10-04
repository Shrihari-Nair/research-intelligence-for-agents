# PDF rendering spike (M17)

Candidates tried on this machine (Ubuntu, Python 3.12):

| Option | Result |
|---|---|
| **WeasyPrint** (HTML + CSS -> PDF) | Works. Unicode text (accents, curly quotes, CJK) renders; page CSS gives a clean report layout. Needs system libs (pango/cairo), present here. |
| ReportLab (programmatic PDF) | Works for Latin text; non-Latin needs manual font registration. Pure Python, no system deps. |
| Headless Chromium (Playwright) | Not used for PDF: heavy install; Playwright is reserved for the optional UI smoke test (M23). |

**Decision:** WeasyPrint, fed with HTML produced from the report's Markdown. It sits behind the `PDFRenderer` interface
(`research_os/artifacts/render_pdf.py`), so ReportLab can replace it on machines without pango. If WeasyPrint cannot import,
the API answers `501` for `format=pdf` with an explanatory message; JSON and Markdown always work.
