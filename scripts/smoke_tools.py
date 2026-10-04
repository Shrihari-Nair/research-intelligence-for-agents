"""Live smoke test for each tool. Prints structured results. Skips tools whose key is missing.

Usage: python -m scripts.smoke_tools [tool ...]    (default: all)
"""
from __future__ import annotations

import os
import sys

from research_os.common import load_config, load_dotenv
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.tools import build_registry
from research_os.tools.registry import ToolError

CASES = {
    "web_search": ({"query": "LangGraph agent framework", "max_results": 3}, "TAVILY_API_KEY"),
    "github_search": ({"query": "langgraph", "max_results": 3}, None),
    "arxiv_search": ({"query": "speculative decoding", "max_results": 3}, None),
    "pubmed_search": ({"query": "migraine prevention", "max_results": 3}, None),
    "fetch_url": ({"url": "https://example.com"}, None),
}


def main() -> None:
    load_dotenv()
    cfg = load_config("configs/tools.yaml")
    ctx = RunContext.create(None, cfg, llm=ScriptedLLM([]), tools=build_registry(cfg, mode="live"))
    wanted = sys.argv[1:] or list(CASES)
    for name in wanted:
        args, env = CASES[name]
        if env and not os.environ.get(env):
            print(f"[{name}] SKIPPED: {env} not set")
            continue
        try:
            res = ctx.tools.call(name, args, ctx)
        except ToolError as e:
            print(f"[{name}] FAILED {e.failure_type.value}: {e}")
            continue
        print(f"[{name}] OK {len(res.sources)} source(s)")
        for s in res.sources:
            print(f"   - {s.source_type:13} {s.title[:70]!r}  {s.url}\n       {s.content[:110]!r}")


if __name__ == "__main__":
    main()
