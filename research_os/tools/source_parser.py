"""HTML/text -> cleaned text, title, published date, source type. Output is UNTRUSTED data."""
from __future__ import annotations

import html as html_lib
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

_SCRIPT = re.compile(r"<(script|style|noscript|template)\b.*?</\1>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_WS = re.compile(r"[ \t\r\f\v]+")


def _strip_html(body: str) -> str:
    text = _TAGS.sub(" ", _SCRIPT.sub(" ", body))
    text = html_lib.unescape(text)
    return re.sub(r"\n\s*\n+", "\n\n", _WS.sub(" ", text)).strip()


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


_DOC_SEGMENTS = {"docs", "doc", "documentation", "tutorial", "tutorials", "guide", "guides", "reference", "manual", "handbook", "learn",
                 "getting-started", "quickstart", "api-reference"}
_DOC_HOSTS = ("readthedocs.io", "readthedocs.org", "palletsprojects.com", "tiangolo.com", "docs.rs", "mozilla.org")
_DOC_TITLE = re.compile(r"\b(documentation|docs|api reference|user guide|developer guide|reference guide|manual|tutorial)\b", re.I)


def classify_source_type(url: str, title: str = "") -> str:
    """Heuristic Source.source_type from the URL (and, for ambiguous hosts, the page title)."""
    p = urlsplit(url)
    host, path = (p.hostname or "").lower(), p.path.lower()
    segs = {x for x in path.split("/") if x}
    if host in ("github.com", "gitlab.com"):
        return "repository"
    if host.endswith(("arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "openreview.net", "aclanthology.org")):
        return "paper"
    if host.endswith(("stackoverflow.com", "reddit.com", "news.ycombinator.com", "discourse.org")):
        return "forum"
    if host.endswith(("medium.com", "substack.com", "dev.to", "hashnode.dev")) or host.startswith("blog.") or "blog" in segs:
        return "blog"
    if host.endswith(("techcrunch.com", "reuters.com", "theverge.com", "bloomberg.com", "wired.com", "zdnet.com")) or "news" in segs:
        return "news"
    if (host.startswith(("docs.", "doc.", "developer.", "developers.")) or host.endswith(_DOC_HOSTS) or segs & _DOC_SEGMENTS
            or _DOC_TITLE.search(title or "")):
        return "documentation"
    return "other"


def parse_document(body: str, content_type: str, max_chars: int) -> tuple[str, str, datetime | None]:
    """Return (title, text, published_at). Text is capped at `max_chars`."""
    if "html" not in content_type.lower():
        text = _WS.sub(" ", body).strip()
        return "", text[:max_chars], None
    title, published, text = "", None, None
    m = _TITLE.search(body)
    if m:
        title = re.sub(r"\s+", " ", html_lib.unescape(m.group(1))).strip()
    try:
        import trafilatura

        text = trafilatura.extract(body, include_comments=False, include_tables=True, favor_recall=True)
        meta = trafilatura.extract_metadata(body)
        if meta is not None:
            title = title or (meta.title or "")
            published = _parse_date(meta.date)
    except Exception:  # extraction is best effort; fall back to tag stripping
        text = None
    if not text or len(text) < 20:
        text = _strip_html(body)
    return title, text.strip()[:max_chars], published
