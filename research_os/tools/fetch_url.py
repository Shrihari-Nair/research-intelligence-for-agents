"""fetch_url: safe HTTP fetch -> cleaned Source. SSRF-protected (also across redirects), size/time/type limited."""
from __future__ import annotations

import ipaddress
import socket
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import httpx

from research_os.common import load_config
from research_os.contracts import FailureType, Source, canonical_url, source_id_for_url
from research_os.tools.registry import ToolError, ToolResult
from research_os.tools.source_parser import classify_source_type, parse_document

ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/xml", "text/xml")


def _blocked(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified or (isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10")))


def check_url_safe(url: str, resolver=socket.getaddrinfo) -> str:
    """Raise ToolError(INVALID_SOURCE) unless `url` is http(s) and every resolved address is public. Returns the vetted IP to connect to."""
    p = urlsplit(url.strip())
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ToolError(FailureType.INVALID_SOURCE, f"only http/https URLs are allowed: {url!r}")
    host = p.hostname
    try:
        addrs = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = resolver(host, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except OSError as e:
            raise ToolError(FailureType.INVALID_SOURCE, f"cannot resolve host {host!r}: {e}") from e
        addrs = [ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos]
    if not addrs or any(_blocked(a) for a in addrs):
        raise ToolError(FailureType.INVALID_SOURCE, f"blocked address for host {host!r} (private/loopback/link-local)")
    return str(addrs[0])


class PinningTransport(httpx.BaseTransport):
    """Connects to the IP that passed `check_url_safe` (request extension `pin_ip`) instead of resolving the host again.

    Closes the DNS-rebinding window between the safety check and the connection. The Host header keeps the original name and TLS
    verification / SNI use it too (`sni_hostname`), so certificates are still checked against the real host.
    """

    def __init__(self, inner: httpx.BaseTransport | None = None):
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        ip = request.extensions.pop("pin_ip", None)
        host = request.url.host
        if ip and host != ip:
            request.extensions["sni_hostname"] = host
            request.url = request.url.copy_with(host=ip)
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


def fetch_url_tool(args: dict, ctx, *, cfg: dict | None = None, client: httpx.Client | None = None,
                   resolver=socket.getaddrinfo) -> ToolResult:
    """Fetch one URL and return it as a Source."""
    cfg = cfg or load_config("configs/tools.yaml")
    h, limits = cfg["http"], cfg["limits"]
    url = args.get("url")
    if not url or not isinstance(url, str):
        raise ToolError(FailureType.INVALID_SOURCE, "fetch_url requires a 'url' string")
    own = client is None
    client = client or httpx.Client(trust_env=False, transport=PinningTransport())   # no proxy env vars, no second DNS lookup
    try:
        current = url
        for hop in range(h["max_redirects"] + 1):
            ip = check_url_safe(current, resolver)
            with client.stream("GET", current, headers={"User-Agent": h["user_agent"]}, timeout=h["timeout_s"],
                               follow_redirects=False, extensions={"pin_ip": ip}) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    if hop == h["max_redirects"]:
                        raise ToolError(FailureType.INVALID_SOURCE, f"too many redirects (> {h['max_redirects']})")
                    current = urljoin(current, r.headers["location"])
                    continue
                if r.status_code >= 500 or r.status_code == 429:
                    raise ToolError(FailureType.TOOL_FAILURE, f"HTTP {r.status_code} from {current}")
                if r.status_code >= 400:
                    raise ToolError(FailureType.INVALID_SOURCE, f"HTTP {r.status_code} from {current}")
                ctype = r.headers.get("content-type", "text/plain").split(";")[0].strip().lower()
                if ctype not in ALLOWED_TYPES:
                    raise ToolError(FailureType.INVALID_SOURCE, f"unsupported content type {ctype!r}")
                buf, size = [], 0
                for chunk in r.iter_bytes():
                    size += len(chunk)
                    if size > h["max_bytes"]:
                        raise ToolError(FailureType.INVALID_SOURCE, f"response too large (> {h['max_bytes']} bytes)")
                    buf.append(chunk)
                charset = r.charset_encoding or "utf-8"
                body = b"".join(buf).decode(charset, errors="replace")
                break
        title, text, published = parse_document(body, ctype, limits["max_content_chars"])
        if not text:
            raise ToolError(FailureType.INVALID_SOURCE, "no extractable text")
        final = canonical_url(current)
        src = Source(source_id=source_id_for_url(current), title=title or urlsplit(final).netloc, url=final,
                     source_type=classify_source_type(final, title), published_at=published,
                     retrieved_at=datetime.now(timezone.utc), content=text, tool="fetch_url")
        return ToolResult(sources=[src])
    except httpx.TimeoutException as e:
        raise ToolError(FailureType.SEARCH_TIMEOUT, f"timeout fetching {url}: {e}") from e
    except httpx.HTTPError as e:
        raise ToolError(FailureType.TOOL_FAILURE, f"{type(e).__name__} fetching {url}: {e}") from e
    finally:
        if own:
            client.close()
