import pytest

from research_os.router.entities import EntityExtractor


@pytest.fixture(scope="module")
def ex():
    return EntityExtractor.from_config()


def test_compare_langgraph_and_crewai(ex):
    e = ex.extract("Compare LangGraph and CrewAI")
    assert e["technologies"] == ["LangGraph", "CrewAI"]
    assert e["companies"] == [] and e["time_scope"] == []


def test_recent_papers_extracts_topic_and_recency(ex):
    e = ex.extract("Find recent papers on small speech models")
    assert e["topics"] == ["small speech models"] and e["time_scope"] == ["recent"]


def test_spoken_lowercase_and_aliases(ex):
    e = ex.extract("um compare langgraph and crew ai and also postgres")
    assert e["technologies"] == ["LangGraph", "CrewAI", "PostgreSQL"]


def test_companies_and_since_year(ex):
    e = ex.extract("What has Anthropic launched since 2024?")
    assert e["companies"] == ["Anthropic"] and e["time_scope"] == ["since 2024"]


def test_quoted_strings_become_topics(ex):
    assert "agent memory" in ex.extract('track "agent memory" research for me')["topics"]


def test_unknown_camelcase_token_is_a_technology_candidate(ex):
    assert ex.extract("how does FooBarDB handle sharding")["technologies"] == ["FooBarDB"]


def test_sentence_initial_capital_word_is_not_an_entity(ex):
    e = ex.extract("Tell me about the market")
    assert e["technologies"] == [] and e["companies"] == []


def test_time_scope_variants(ex):
    assert ex.extract("what's the latest on rag")["time_scope"] == ["latest"]
    assert ex.extract("developments over the last 2 years")["time_scope"] == ["last 2 years"]
    assert ex.extract("papers in 2023")["time_scope"] == ["in 2023"]
    assert ex.extract("who created fastapi")["time_scope"] == []


def test_dedupes_and_keeps_first_seen_order(ex):
    assert ex.extract("langgraph vs LangGraph vs crewai")["technologies"] == ["LangGraph", "CrewAI"]


def test_empty_and_symbols_do_not_crash(ex):
    for t in ["", "   ", "?!?", "a" * 5000]:
        e = ex.extract(t)
        assert set(e) == {"technologies", "companies", "topics", "time_scope"}
