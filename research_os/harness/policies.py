"""Central policy checks (domains, permissions live in the registry; evidence requirements live in workflows.yaml)."""
from __future__ import annotations

from urllib.parse import urlsplit

from research_os.contracts import FailureType
from research_os.tools.registry import ToolError


def _matches(host: str, domain: str) -> bool:
    d = domain.lower().lstrip(".")
    return host == d or host.endswith("." + d)


def check_fetch_policy(url: str, policy: dict) -> None:
    """Raise ToolError(INVALID_SOURCE) if the URL's host is blocked or not on the allow list."""
    host = (urlsplit(url).hostname or "").lower()
    if any(_matches(host, d) for d in policy.get("blocked_domains", [])):
        raise ToolError(FailureType.INVALID_SOURCE, f"domain policy: {host} is blocked")
    allowed = policy.get("allowed_domains", [])
    if allowed and not any(_matches(host, d) for d in allowed):
        raise ToolError(FailureType.INVALID_SOURCE, f"domain policy: {host} is not on the allowed list")
