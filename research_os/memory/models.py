"""SQLAlchemy models for persistent research memory (SQLite by default, Postgres via DATABASE_URL)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base."""


class RunRow(Base):
    __tablename__ = "research_runs"
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="RUNNING")
    workflow: Mapped[str | None] = mapped_column(String(64), nullable=True)
    llm_backend: Mapped[str] = mapped_column(String(24), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    budget: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    router: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class QuestionRow(Base):
    __tablename__ = "research_questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.run_id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)


class SourceRow(Base):
    __tablename__ = "sources"
    source_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(24))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    tool: Mapped[str] = mapped_column(String(32))


class RunSourceRow(Base):
    __tablename__ = "run_sources"
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.run_id", ondelete="CASCADE"), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), primary_key=True)


class ClaimRow(Base):
    __tablename__ = "claims"
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.run_id", ondelete="CASCADE"), primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    claim: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(String(32), index=True)
    evidence_quote: Mapped[str] = mapped_column(Text)
    extractor_confidence: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(16), index=True)
    verdict_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)


class EvidenceRow(Base):
    __tablename__ = "evidence"
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(32))
    quote: Mapped[str] = mapped_column(Text)
    quote_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)


class ReportRow(Base):
    __tablename__ = "reports"
    run_id: Mapped[str] = mapped_column(ForeignKey("research_runs.run_id", ondelete="CASCADE"), primary_key=True)
    content: Mapped[dict] = mapped_column(JSON)
    summary: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
