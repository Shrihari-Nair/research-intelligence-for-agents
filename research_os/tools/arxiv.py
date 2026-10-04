"""arxiv_search: public arXiv Atom API -> paper Sources (abstract text)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from research_os.common import load_config
from research_os.contracts import FailureType
from research_os.tools._common import build_source, clamp_results, http_client, parse_dt, request, require_query
from research_os.tools.registry import ToolError, ToolResult

NS = {"a": "http://www.w3.org/2005/Atom"}


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def arxiv_search_tool(args: dict, ctx, *, cfg: dict | None = None, client: httpx.Client | None = None) -> ToolResult:
    """Search arXiv by relevance (args.sort == 'recent' for newest first). Query words are ANDed."""
    cfg = cfg or load_config("configs/tools.yaml")
    query, n = require_query(args), clamp_results(args, cfg)
    terms = [re.sub(r"[^\w\-]", "", w) for w in query.split()]
    search = "+AND+".join(f"all:{t}" for t in terms if t) or f"all:{query}"
    with http_client(client) as c:
        sort_by = "submittedDate" if args.get("sort") == "recent" else "relevance"
        r = request(c, "GET", cfg["arxiv"]["endpoint"], cfg,
                    params={"search_query": search.replace("+", " "), "max_results": n, "sortBy": sort_by,
                            "sortOrder": "descending"})
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        raise ToolError(FailureType.TOOL_FAILURE, f"arXiv returned invalid XML: {e}") from e
    out = []
    for e in root.findall("a:entry", NS):
        url, title, summary = _clean(e.findtext("a:id", namespaces=NS)), _clean(e.findtext("a:title", namespaces=NS)), \
            _clean(e.findtext("a:summary", namespaces=NS))
        if not url or not summary:
            continue
        authors = ", ".join(_clean(a.findtext("a:name", namespaces=NS)) for a in e.findall("a:author", NS))
        out.append(build_source(url=url.replace("http://", "https://", 1), title=title,
                                content=f"{title}. Authors: {authors}. Abstract: {summary}", tool="arxiv_search",
                                source_type="paper", published_at=parse_dt(e.findtext("a:published", namespaces=NS)), cfg=cfg))
    return ToolResult(sources=out)
