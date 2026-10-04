"""Shared helpers for search tools."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import httpx

from research_os.contracts import FailureType, Source, canonical_url, source_id_for_url
from research_os.tools.registry import ToolError


@contextmanager
def http_client(client: httpx.Client | None):
    """Yield the injected client or a fresh one (closed afterwards)."""
    if client is not None:
        yield client
        return
    c = httpx.Client()
    try:
        yield c
    finally:
        c.close()


def request(client: httpx.Client, method: str, url: str, cfg: dict, **kw) -> httpx.Response:
    """HTTP request with failures mapped to ToolError (timeouts -> SEARCH_TIMEOUT, everything else -> TOOL_FAILURE)."""
    headers = {"User-Agent": cfg["http"]["user_agent"], **kw.pop("headers", {})}
    try:
        r = client.request(method, url, headers=headers, timeout=cfg["http"]["timeout_s"], **kw)
    except httpx.TimeoutException as e:
        raise ToolError(FailureType.SEARCH_TIMEOUT, f"timeout calling {url}: {e}") from e
    except httpx.HTTPError as e:
        raise ToolError(FailureType.TOOL_FAILURE, f"{type(e).__name__} calling {url}: {e}") from e
    if r.status_code >= 400:
        detail = r.text[:200].replace("\n", " ")
        raise ToolError(FailureType.TOOL_FAILURE, f"HTTP {r.status_code} from {url}: {detail}")
    return r


def clamp_results(args: dict, cfg: dict, default: int = 5) -> int:
    """max_results bounded by config."""
    cap = cfg["limits"]["max_results"]
    try:
        n = int(args.get("max_results", default))
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, cap))


def require_query(args: dict) -> str:
    """The non-empty 'query' string or ToolError."""
    q = args.get("query")
    if not isinstance(q, str) or not q.strip():
        raise ToolError(FailureType.TOOL_FAILURE, "tool requires a non-empty 'query' string")
    return q.strip()


def parse_dt(value) -> datetime | None:
    """Best-effort ISO date -> aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def build_source(*, url: str, title: str, content: str, tool: str, source_type: str, published_at, cfg: dict) -> Source:
    """Source with canonical URL, stable id and size-capped content."""
    return Source(source_id=source_id_for_url(url), title=title.strip() or url, url=canonical_url(url),
                  source_type=source_type, published_at=published_at, retrieved_at=datetime.now(timezone.utc),
                  content=content.strip()[:cfg["limits"]["max_content_chars"]], tool=tool)
