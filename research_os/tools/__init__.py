"""Research tools. Build a registry for live or fixture mode with `build_registry`."""
from __future__ import annotations

import os

from research_os.common import load_config
from research_os.harness.policies import check_fetch_policy
from research_os.tools._common import clamp_results, require_query
from research_os.tools.arxiv import arxiv_search_tool
from research_os.tools.fetch_url import fetch_url_tool
from research_os.tools.fixtures import FixtureCorpus
from research_os.tools.github import github_search_tool
from research_os.tools.pubmed import pubmed_search_tool
from research_os.tools.registry import ToolRegistry, ToolResult, ToolSpec
from research_os.tools.schemas import FetchInput, MemorySearchInput, ScholarlySearchInput, SearchInput
from research_os.tools.web_search import web_search_tool

SEARCH_TOOLS = ("web_search", "github_search", "arxiv_search", "pubmed_search")


def build_registry(cfg: dict | None = None, mode: str | None = None, memory=None, corpus=None) -> ToolRegistry:
    """Registry with all five tools. mode 'live' hits real APIs; 'fixture' serves the synthetic corpus."""
    cfg = cfg or load_config("configs/tools.yaml")
    mode = mode or os.environ.get("RESEARCH_OS_TOOLS") or cfg["mode"]
    if mode not in ("live", "fixture"):
        raise ValueError(f"unknown tool mode {mode!r} (expected 'live' or 'fixture')")
    reg_cfg = cfg.get("registry", {})
    reg = ToolRegistry(retry_cfg=reg_cfg, permissions=cfg.get("permissions"), default_timeout=reg_cfg.get("default_timeout_s", 15))
    cap = cfg["limits"]["max_content_chars"]
    schemas = {"web_search": SearchInput, "github_search": SearchInput, "arxiv_search": ScholarlySearchInput,
               "pubmed_search": ScholarlySearchInput, "fetch_url": FetchInput}
    if mode == "live":
        fns = {"web_search": web_search_tool, "github_search": github_search_tool, "arxiv_search": arxiv_search_tool,
               "pubmed_search": pubmed_search_tool, "fetch_url": fetch_url_tool}
        impl = {n: (lambda args, ctx, fn=fn: fn(args, ctx, cfg=cfg)) for n, fn in fns.items()}
    else:
        corpus = corpus or FixtureCorpus.load(cfg["fixtures_dir"])  # any object with search(query, tool, k) and fetch(url)
        impl = {n: (lambda args, ctx, n=n: ToolResult(sources=corpus.search(require_query(args), n, clamp_results(args, cfg))))
                for n in SEARCH_TOOLS}
        impl["fetch_url"] = lambda args, ctx: ToolResult(sources=[corpus.fetch(str(args.get("url", "")))])
    policy = cfg.get("policy", {})
    for name, fn in impl.items():
        reg.register_spec(ToolSpec(
            name=name, fn=fn, cost_class="search" if name in SEARCH_TOOLS else "other", input_schema=schemas[name],
            permissions=("research",), max_content_chars=cap,
            pre_checks=(lambda a, p=policy: check_fetch_policy(a["url"], p),) if name == "fetch_url" else ()))
    if memory is not None:
        reg.register_spec(ToolSpec(
            name="memory_search", cost_class="other", input_schema=MemorySearchInput, permissions=("memory",),
            fn=lambda args, ctx: ToolResult(data={"results": memory.search(args["query"], tuple(args["kinds"]), args["k"], args["min_similarity"])})))
    return reg
