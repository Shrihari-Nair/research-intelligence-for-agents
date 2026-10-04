"""github_search: GitHub REST repository search -> repository Sources (metadata text, no code)."""
from __future__ import annotations

import os

import httpx

from research_os.common import load_config
from research_os.tools._common import build_source, clamp_results, http_client, parse_dt, request, require_query
from research_os.tools.registry import ToolResult


def github_search_tool(args: dict, ctx, *, cfg: dict | None = None, client: httpx.Client | None = None) -> ToolResult:
    """Search repositories by query; one Source per repo with description, language, license, stars, last push."""
    cfg = cfg or load_config("configs/tools.yaml")
    query, n = require_query(args), clamp_results(args, cfg)
    g = cfg["github"]
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get(g["token_env"])
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with http_client(client) as c:
        r = request(c, "GET", f"{g['endpoint']}/search/repositories", cfg, headers=headers,
                    params={"q": query, "per_page": n, "sort": "stars", "order": "desc"})
    out = []
    for it in r.json().get("items", []):
        if not it.get("html_url") or not it.get("full_name"):
            continue
        lic = (it.get("license") or {}).get("spdx_id") or "none"
        content = (f"{it['full_name']}: {it.get('description') or 'no description'}. Language: {it.get('language') or 'unknown'}. "
                   f"License: {lic}. Stars: {it.get('stargazers_count', 0)}. Last pushed: {it.get('pushed_at') or 'unknown'}. "
                   f"Topics: {', '.join(it.get('topics') or []) or 'none'}.")
        out.append(build_source(url=it["html_url"], title=it["full_name"], content=content, tool="github_search",
                                source_type="repository", published_at=parse_dt(it.get("pushed_at")), cfg=cfg))
    return ToolResult(sources=out)
