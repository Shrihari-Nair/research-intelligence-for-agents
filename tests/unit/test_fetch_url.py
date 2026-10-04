import httpx
import pytest

from research_os.contracts import FailureType
from research_os.tools.fetch_url import check_url_safe, fetch_url_tool
from research_os.tools.registry import ToolError

CFG = {"http": {"timeout_s": 5, "max_redirects": 3, "max_bytes": 2000, "user_agent": "UA/1"},
       "limits": {"max_content_chars": 500}}


def resolver(mapping):
    def r(host, port, *a, **k):
        ips = mapping.get(host)
        if ips is None:
            raise OSError("no such host")
        return [(2, 1, 6, "", (ip, port)) for ip in ips]
    return r


PUBLIC = resolver({"example.com": ["93.184.216.34"], "other.com": ["93.184.216.35"],
                   "evil.com": ["10.0.0.5"], "meta.com": ["169.254.169.254"], "loop.com": ["127.0.0.1"],
                   "v6.com": ["::1"], "mapped.com": ["::ffff:127.0.0.1"], "ll6.com": ["fe80::1"]})


@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)", "http://", "example.com"])
def test_bad_schemes_blocked(url):
    with pytest.raises(ToolError) as e:
        check_url_safe(url, PUBLIC)
    assert e.value.failure_type == FailureType.INVALID_SOURCE


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost/", "http://10.1.2.3/", "http://192.168.0.1/", "http://169.254.169.254/latest/",
    "http://[::1]/", "http://0.0.0.0/", "http://evil.com/", "http://meta.com/", "http://loop.com/", "http://v6.com/",
    "http://mapped.com/", "http://ll6.com/", "http://2130706433/", "http://nonexistent.invalid/"])
def test_private_loopback_linklocal_and_unresolvable_blocked(url):
    with pytest.raises(ToolError) as e:
        check_url_safe(url, PUBLIC)
    assert e.value.failure_type == FailureType.INVALID_SOURCE


def test_public_url_allowed():
    check_url_safe("https://example.com/a", PUBLIC)


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def html(body, ctype="text/html; charset=utf-8", status=200, headers=None):
    return httpx.Response(status, content=body.encode(), headers={"content-type": ctype, **(headers or {})})


def run(handler, url="https://example.com/p", cfg=CFG):
    return fetch_url_tool({"url": url}, None, cfg=cfg, client=client(handler), resolver=PUBLIC)


def test_fetch_returns_cleaned_source_with_stable_id():
    page = "<html><head><title>My Doc</title></head><body><main><h1>Hi</h1><p>LangGraph is a library for agents. " \
           "It builds stateful graphs of steps for language model apps.</p></main><script>evil()</script></body></html>"
    res = run(lambda r: html(page))
    s = res.sources[0]
    assert s.title == "My Doc" and "LangGraph is a library" in s.content and "evil()" not in s.content
    assert s.tool == "fetch_url" and s.url == "https://example.com/p" and len(s.source_id) == 16


def test_redirect_to_private_address_is_blocked():
    def h(r):
        if r.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://evil.com/secret"})
        return html("secret")
    with pytest.raises(ToolError) as e:
        run(h)
    assert e.value.failure_type == FailureType.INVALID_SOURCE


def test_redirect_limit_enforced():
    with pytest.raises(ToolError) as e:
        run(lambda r: httpx.Response(302, headers={"location": "https://example.com/loop"}))
    assert "redirect" in str(e.value).lower()


def test_safe_redirect_is_followed():
    def h(r):
        if r.url.path == "/p":
            return httpx.Response(301, headers={"location": "https://other.com/final"})
        return html("<html><body><p>Final destination page with enough text to be extracted properly here.</p></body></html>")
    assert "Final destination" in run(h).sources[0].content


def test_oversized_body_is_rejected():
    with pytest.raises(ToolError) as e:
        run(lambda r: html("<p>" + "x" * 5000 + "</p>"))
    assert "too large" in str(e.value).lower()


def test_non_text_content_type_rejected():
    with pytest.raises(ToolError) as e:
        run(lambda r: html("PNG....", ctype="image/png"))
    assert e.value.failure_type == FailureType.INVALID_SOURCE


def test_http_error_status_is_invalid_source_and_5xx_is_tool_failure():
    with pytest.raises(ToolError) as e:
        run(lambda r: html("nope", status=404))
    assert e.value.failure_type == FailureType.INVALID_SOURCE
    with pytest.raises(ToolError) as e:
        run(lambda r: html("boom", status=503))
    assert e.value.failure_type == FailureType.TOOL_FAILURE


def test_timeout_is_search_timeout():
    def h(r):
        raise httpx.ReadTimeout("slow")
    with pytest.raises(ToolError) as e:
        run(h)
    assert e.value.failure_type == FailureType.SEARCH_TIMEOUT


def test_content_is_capped_and_user_agent_sent():
    seen = {}

    def h(r):
        seen["ua"] = r.headers["user-agent"]
        return html("<html><body><p>" + "word " * 1000 + "</p></body></html>")
    cfg = {**CFG, "http": {**CFG["http"], "max_bytes": 100000}}
    res = run(h, cfg=cfg)
    assert len(res.sources[0].content) <= 500 and seen["ua"] == "UA/1"


def test_plain_text_is_accepted():
    res = run(lambda r: html("plain text body with several words in it", ctype="text/plain"))
    assert "plain text body" in res.sources[0].content


# ---- DNS rebinding: connect to the address that passed the check ----
def test_the_connection_goes_to_the_vetted_ip_even_if_dns_would_now_answer_differently():
    from research_os.tools.fetch_url import PinningTransport
    seen = {}
    calls = {"n": 0}

    def rebinding_resolver(host, port, *a, **k):          # public on the first lookup, loopback on any later one
        calls["n"] += 1
        ip = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"
        return [(2, 1, 6, "", (ip, port))]

    def handler(request):
        seen.update(host=request.url.host, host_header=request.headers["host"], sni=request.extensions.get("sni_hostname"))
        return html("<html><body><main><p>LangGraph is a library for building agents with graphs and state.</p></main></body></html>")

    c = httpx.Client(transport=PinningTransport(httpx.MockTransport(handler)))
    fetch_url_tool({"url": "https://example.com/p"}, None, cfg=CFG, client=c, resolver=rebinding_resolver)
    assert seen["host"] == "93.184.216.34" and seen["host_header"] == "example.com" and seen["sni"] == "example.com"


def test_ip_literal_urls_are_not_rewritten():
    from research_os.tools.fetch_url import PinningTransport
    seen = {}
    def handler(r):
        seen.update(host=r.url.host)
        return html("<p>ok text here for the page body content</p>")

    c = httpx.Client(transport=PinningTransport(httpx.MockTransport(handler)))
    fetch_url_tool({"url": "http://93.184.216.34/x"}, None, cfg=CFG, client=c, resolver=PUBLIC)
    assert seen["host"] == "93.184.216.34"
