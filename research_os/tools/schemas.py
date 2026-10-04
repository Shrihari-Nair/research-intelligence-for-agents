"""Pydantic input schemas for every tool (validated by the registry before the tool runs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """web_search / github_search."""

    query: str = Field(min_length=1, max_length=400)
    max_results: int = Field(default=5, ge=1, le=20)


class ScholarlySearchInput(SearchInput):
    """arxiv_search / pubmed_search."""

    sort: Literal["relevance", "recent"] = "relevance"


class FetchInput(BaseModel):
    """fetch_url."""

    url: str = Field(min_length=1, max_length=2048)


class MemorySearchInput(BaseModel):
    """memory_search."""

    query: str = Field(min_length=1, max_length=400)
    kinds: list[Literal["questions", "claims", "reports"]] = Field(default_factory=lambda: ["questions", "claims", "reports"])
    k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0)
