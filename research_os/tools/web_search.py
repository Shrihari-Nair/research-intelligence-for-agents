"""web_search: pluggable provider (Tavily implemented). Key from the environment only."""
from __future__ import annotations

import os

import httpx

from research_os.common import load_config
from research_os.contracts import FailureType
from research_os.tools._common import build_source, clamp_results, http_client, parse_dt, request, require_query
from research_os.tools.registry import ToolError, ToolResult
from research_os.tools.source_parser import classify_source_type


def _tavily(query: str, n: int, cfg: dict, client: httpx.Client) -> list[dict]:
    ws = cfg["web_search"]
    key = os.environ.get(ws["api_key_env"])
    if not key:
        raise ToolError(FailureType.TOOL_FAILURE, f"{ws['api_key_env']} is not set; web_search (tavily) is unavailable")
    r = request(client, "POST", ws["endpoint"], cfg, headers={"Authorization": f"Bearer {key}"},
                json={"query": query, "max_results": n, "search_depth": "basic", "include_answer": False,
                      "include_published_date": True})
    try:
        return list(r.json().get("results", []))
    except ValueError as e:
        raise ToolError(FailureType.TOOL_FAILURE, f"tavily returned invalid JSON: {e}") from e


PROVIDERS = {"tavily": _tavily}


def web_search_tool(args: dict, ctx, *, cfg: dict | None = None, client: httpx.Client | None = None) -> ToolResult:
    """Search the web through the configured provider; results become Sources (snippets, deduplicated by URL)."""
    cfg = cfg or load_config("configs/tools.yaml")
    query, n = require_query(args), clamp_results(args, cfg)
    provider = PROVIDERS.get(cfg["web_search"]["provider"])
    if provider is None:
        raise ToolError(FailureType.TOOL_FAILURE, f"unknown web_search provider {cfg['web_search']['provider']!r}")
    with http_client(client) as c:
        raw = provider(query, n, cfg, c)
    out, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or not item.get("url") or not item.get("content"):
            continue  # a result without a URL or text cannot be cited
        src = build_source(url=item["url"], title=item.get("title") or "", content=item["content"], tool="web_search",
                           source_type=classify_source_type(item["url"], item.get("title") or ""), published_at=parse_dt(item.get("published_date")),
                           cfg=cfg)
        if src.source_id not in seen:
            seen.add(src.source_id)
            out.append(src)
    return ToolResult(sources=out)
