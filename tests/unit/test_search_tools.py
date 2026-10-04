import json

import httpx
import pytest

from research_os.contracts import FailureType
from research_os.tools.arxiv import arxiv_search_tool
from research_os.tools.github import github_search_tool
from research_os.tools.pubmed import pubmed_search_tool
from research_os.tools.registry import ToolError
from research_os.tools.web_search import web_search_tool

CFG = {
    "http": {"timeout_s": 5, "max_redirects": 3, "max_bytes": 100000, "user_agent": "UA/1"},
    "limits": {"max_content_chars": 400, "max_results": 5},
    "web_search": {"provider": "tavily", "api_key_env": "TAVILY_API_KEY", "endpoint": "https://api.tavily.com/search"},
    "github": {"token_env": "GITHUB_TOKEN", "endpoint": "https://api.github.com"},
    "arxiv": {"endpoint": "https://export.arxiv.org/api/query"},
    "pubmed": {"endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", "api_key_env": "NCBI_API_KEY"},
}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------- web_search (Tavily) ----------------
def test_tavily_request_shape_and_parsing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    seen = {}

    def h(r):
        seen["auth"], seen["body"], seen["url"] = r.headers["authorization"], json.loads(r.content), str(r.url)
        return httpx.Response(200, json={"results": [
            {"title": "LangGraph docs", "url": "https://docs.langchain.com/langgraph?utm_source=x",
             "content": "LangGraph builds graphs.", "score": 0.9, "published_date": "2025-03-01"},
            {"title": "Dup", "url": "https://docs.langchain.com/langgraph", "content": "again", "score": 0.5}]})
    res = web_search_tool({"query": "langgraph", "max_results": 2}, None, cfg=CFG, client=client(h))
    assert seen["auth"] == "Bearer tvly-test" and seen["body"]["query"] == "langgraph" and seen["body"]["max_results"] == 2
    assert seen["url"] == "https://api.tavily.com/search"
    assert len(res.sources) == 1  # same canonical URL deduplicated
    s = res.sources[0]
    assert s.tool == "web_search" and s.source_type == "documentation" and s.published_at.year == 2025
    assert s.content == "LangGraph builds graphs."


def test_tavily_missing_key_is_a_clear_tool_failure(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ToolError, match="TAVILY_API_KEY"):
        web_search_tool({"query": "x"}, None, cfg=CFG, client=client(lambda r: httpx.Response(200, json={})))


def test_web_search_timeout_and_http_errors(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")

    def slow(r):
        raise httpx.ConnectTimeout("t")
    with pytest.raises(ToolError) as e:
        web_search_tool({"query": "x"}, None, cfg=CFG, client=client(slow))
    assert e.value.failure_type == FailureType.SEARCH_TIMEOUT
    with pytest.raises(ToolError) as e:
        web_search_tool({"query": "x"}, None, cfg=CFG, client=client(lambda r: httpx.Response(500, text="x")))
    assert e.value.failure_type == FailureType.TOOL_FAILURE


def test_web_search_requires_query_and_clamps_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    with pytest.raises(ToolError):
        web_search_tool({}, None, cfg=CFG, client=client(lambda r: httpx.Response(200, json={})))
    seen = {}

    def h(r):
        seen["n"] = json.loads(r.content)["max_results"]
        return httpx.Response(200, json={"results": []})
    web_search_tool({"query": "x", "max_results": 999}, None, cfg=CFG, client=client(h))
    assert seen["n"] == 5


def test_garbage_results_are_skipped_not_fatal(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    res = web_search_tool({"query": "x"}, None, cfg=CFG, client=client(lambda r: httpx.Response(200, json={"results": [
        {"title": "no url"}, {"url": "https://a.com/ok", "title": "ok", "content": "fine content"}, "junk"]})))
    assert [s.url for s in res.sources] == ["https://a.com/ok"]


# ---------------- github ----------------
GITHUB = {"total_count": 1, "items": [{"full_name": "langchain-ai/langgraph", "html_url": "https://github.com/langchain-ai/langgraph",
          "description": "Build resilient language agents as graphs.", "language": "Python", "stargazers_count": 12345,
          "pushed_at": "2026-09-30T10:00:00Z", "license": {"spdx_id": "MIT"}, "topics": ["agents", "llm"]}]}


def test_github_parses_repo_metadata_into_content():
    seen = {}

    def h(r):
        seen["q"], seen["accept"] = r.url.params["q"], r.headers["accept"]
        return httpx.Response(200, json=GITHUB)
    res = github_search_tool({"query": "langgraph"}, None, cfg=CFG, client=client(h))
    s = res.sources[0]
    assert seen["q"] == "langgraph" and "github" in seen["accept"]
    assert s.source_type == "repository" and s.title == "langchain-ai/langgraph" and s.tool == "github_search"
    for fact in ("MIT", "Python", "12345", "Build resilient language agents as graphs."):
        assert fact in s.content
    assert s.published_at.year == 2026


def test_github_token_sent_when_set_and_rate_limit_is_tool_failure(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    seen = {}

    def h(r):
        seen["auth"] = r.headers.get("authorization")
        return httpx.Response(403, json={"message": "API rate limit exceeded"})
    with pytest.raises(ToolError) as e:
        github_search_tool({"query": "x"}, None, cfg=CFG, client=client(h))
    assert seen["auth"] == "Bearer ghp_x" and e.value.failure_type == FailureType.TOOL_FAILURE
    assert "rate limit" in str(e.value).lower()


def test_github_null_license_and_description_ok():
    item = {**GITHUB["items"][0], "license": None, "description": None}
    res = github_search_tool({"query": "x"}, None, cfg=CFG,
                             client=client(lambda r: httpx.Response(200, json={"items": [item]})))
    assert "License: none" in res.sources[0].content


# ---------------- arxiv ----------------
ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2610.02208v1</id>
    <title>Speculative Decoding
      Revisited</title>
    <summary>We study speculative decoding with draft models.   It speeds up inference.</summary>
    <published>2026-10-01T17:59:58Z</published>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
  </entry>
</feed>"""


def test_arxiv_parses_atom_entries():
    seen = {}

    def h(r):
        seen["q"] = r.url.params["search_query"]
        return httpx.Response(200, text=ATOM, headers={"content-type": "application/atom+xml"})
    res = arxiv_search_tool({"query": "speculative decoding"}, None, cfg=CFG, client=client(h))
    s = res.sources[0]
    assert "speculative" in seen["q"] and "decoding" in seen["q"]
    assert s.source_type == "paper" and s.title == "Speculative Decoding Revisited"
    assert s.url == "https://arxiv.org/abs/2610.02208v1" and "A. Author" in s.content
    assert "It speeds up inference." in s.content and s.published_at.year == 2026


def test_arxiv_empty_feed_returns_no_sources_and_bad_xml_is_tool_failure():
    assert arxiv_search_tool({"query": "x"}, None, cfg=CFG, client=client(
        lambda r: httpx.Response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>"))).sources == []
    with pytest.raises(ToolError) as e:
        arxiv_search_tool({"query": "x"}, None, cfg=CFG, client=client(lambda r: httpx.Response(200, text="<<<not xml")))
    assert e.value.failure_type == FailureType.TOOL_FAILURE


# ---------------- pubmed ----------------
EFETCH = """<?xml version="1.0" ?><PubmedArticleSet><PubmedArticle><MedlineCitation><PMID Version="1">40000001</PMID>
<Article><Journal><JournalIssue><PubDate><Year>2025</Year><Month>Feb</Month></PubDate></JournalIssue><Title>Am J Dent</Title></Journal>
<ArticleTitle>Effect of remineralization product.</ArticleTitle>
<Abstract><AbstractText Label="PURPOSE">To evaluate enamel microhardness.</AbstractText>
<AbstractText Label="RESULTS">Hardness improved.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle><PubmedArticle><MedlineCitation><PMID>40000002</PMID>
<Article><ArticleTitle>No abstract here</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_pubmed_two_step_search_and_fetch():
    calls = []

    def h(r):
        calls.append(r.url.path.rsplit("/", 1)[-1])
        if r.url.path.endswith("esearch.fcgi"):
            assert r.url.params["term"] == "enamel" and r.url.params["retmode"] == "json"
            return httpx.Response(200, json={"esearchresult": {"idlist": ["40000001", "40000002"]}})
        assert r.url.params["id"] == "40000001,40000002"
        return httpx.Response(200, text=EFETCH)
    res = pubmed_search_tool({"query": "enamel"}, None, cfg=CFG, client=client(h))
    assert calls == ["esearch.fcgi", "efetch.fcgi"]
    assert len(res.sources) == 1  # the article without an abstract has nothing to cite and is skipped
    s = res.sources[0]
    assert s.url == "https://pubmed.ncbi.nlm.nih.gov/40000001" and s.source_type == "paper"
    assert "PURPOSE: To evaluate enamel microhardness." in s.content and s.published_at.year == 2025


def test_pubmed_no_hits_skips_second_call():
    calls = []

    def h(r):
        calls.append(1)
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})
    assert pubmed_search_tool({"query": "zzzz"}, None, cfg=CFG, client=client(h)).sources == [] and len(calls) == 1


def test_arxiv_and_pubmed_default_to_relevance_and_support_recent_sort():
    seen = {}

    def arxiv_h(r):
        seen["arxiv"] = r.url.params["sortBy"]
        return httpx.Response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")

    def pubmed_h(r):
        seen.setdefault("pubmed", []).append(r.url.params.get("sort"))
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})
    arxiv_search_tool({"query": "x"}, None, cfg=CFG, client=client(arxiv_h))
    assert seen["arxiv"] == "relevance"
    arxiv_search_tool({"query": "x", "sort": "recent"}, None, cfg=CFG, client=client(arxiv_h))
    assert seen["arxiv"] == "submittedDate"
    pubmed_search_tool({"query": "x"}, None, cfg=CFG, client=client(pubmed_h))
    pubmed_search_tool({"query": "x", "sort": "recent"}, None, cfg=CFG, client=client(pubmed_h))
    assert seen["pubmed"] == [None, "pub_date"]
