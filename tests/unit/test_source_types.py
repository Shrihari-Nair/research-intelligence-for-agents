import pytest

from research_os.tools.source_parser import classify_source_type as c


@pytest.mark.parametrize("url,title,expected", [
    ("https://docs.python.org/3/library/asyncio.html", "", "documentation"),
    ("https://fastapi.tiangolo.com/tutorial/first-steps/", "First Steps - FastAPI", "documentation"),
    ("https://flask.palletsprojects.com/en/stable/", "Welcome to Flask - Flask Documentation (3.1.x)", "documentation"),
    ("https://kubernetes.io/docs/concepts/overview/", "Overview", "documentation"),
    ("https://redis.io/docs/latest/", "Redis docs", "documentation"),
    ("https://www.postgresql.org/docs/current/mvcc.html", "Chapter 13. Concurrency Control", "documentation"),
    ("https://react.dev/reference/react", "React Reference Overview", "documentation"),
    ("https://example.com/", "Project API reference and user guide", "documentation"),
    ("https://github.com/langchain-ai/langgraph", "", "repository"),
    ("https://arxiv.org/abs/2401.00001", "", "paper"),
    ("https://pubmed.ncbi.nlm.nih.gov/123/", "", "paper"),
    ("https://medium.com/@x/some-post", "Why I like Flask", "blog"),
    ("https://www.reddit.com/r/python/comments/abc", "Flask vs FastAPI?", "forum"),
    ("https://stackoverflow.com/questions/1/x", "", "forum"),
    ("https://techcrunch.com/2025/01/01/a/", "A funding round", "news"),
    ("https://www.ibm.com/think/topics/langgraph", "What is LangGraph? | IBM", "other"),
])
def test_classification(url, title, expected):
    assert c(url, title) == expected


def test_title_alone_does_not_override_a_clear_blog_or_forum_host():
    assert c("https://medium.com/x/y", "Documentation done right") == "blog"
    assert c("https://news.ycombinator.com/item?id=1", "Show HN: documentation tool") == "forum"
