"""pubmed_search: NCBI E-utilities (esearch + efetch) -> paper Sources with abstracts."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from research_os.common import load_config
from research_os.contracts import FailureType
from research_os.tools._common import build_source, clamp_results, http_client, request, require_query
from research_os.tools.registry import ToolError, ToolResult

_MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _pubdate(article: ET.Element) -> datetime | None:
    y = article.findtext(".//JournalIssue/PubDate/Year")
    if not y or not y.isdigit():
        return None
    m = article.findtext(".//JournalIssue/PubDate/Month") or "1"
    month = int(m) if m.isdigit() else _MONTHS.get(m[:3], 1)
    return datetime(int(y), month, 1, tzinfo=timezone.utc)


def pubmed_search_tool(args: dict, ctx, *, cfg: dict | None = None, client: httpx.Client | None = None) -> ToolResult:
    """Search PubMed by relevance (args.sort == 'recent': newest first). Only articles with an abstract become Sources."""
    cfg = cfg or load_config("configs/tools.yaml")
    query, n = require_query(args), clamp_results(args, cfg)
    p = cfg["pubmed"]
    key = os.environ.get(p["api_key_env"])
    extra = {"api_key": key} if key else {}
    order = {"sort": "pub_date"} if args.get("sort") == "recent" else {}
    with http_client(client) as c:
        s = request(c, "GET", f"{p['endpoint']}/esearch.fcgi", cfg,
                    params={"db": "pubmed", "term": query, "retmax": n, "retmode": "json", **order, **extra})
        ids = s.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ToolResult(sources=[])
        f = request(c, "GET", f"{p['endpoint']}/efetch.fcgi", cfg,
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract", **extra})
    try:
        root = ET.fromstring(f.text)
    except ET.ParseError as e:
        raise ToolError(FailureType.TOOL_FAILURE, f"PubMed returned invalid XML: {e}") from e
    out = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        parts = []
        for at in art.findall(".//Abstract/AbstractText"):
            text = "".join(at.itertext()).strip()
            if text:
                parts.append(f"{at.get('Label')}: {text}" if at.get("Label") else text)
        if not pmid or not parts:
            continue
        title = (art.findtext(".//ArticleTitle") or "").strip()
        out.append(build_source(url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", title=title,
                                content=f"{title} " + " ".join(parts), tool="pubmed_search", source_type="paper",
                                published_at=_pubdate(art), cfg=cfg))
    return ToolResult(sources=out)
