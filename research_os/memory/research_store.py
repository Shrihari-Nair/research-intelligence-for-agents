"""ResearchStore: repository for runs, sources, claims, evidence, reports and similarity search over past work."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from research_os.agents.text import find_quote
from research_os.contracts import Source
from research_os.memory import models as m


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


class ResearchStore:
    """SQLAlchemy-backed store. `url` is any SQLAlchemy URL (sqlite:///file.db, sqlite://, postgresql+psycopg://...)."""

    def __init__(self, url: str, embedder=None):
        kw: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kw["poolclass"] = StaticPool
        self.url = url
        self.engine = create_engine(url, future=True, **kw)
        self._Session = sessionmaker(self.engine, expire_on_commit=False)
        if embedder is None:
            from research_os.memory.embeddings import SentenceEmbedder
            embedder = SentenceEmbedder()
        self.embedder = embedder

    # ---- schema ----
    def create_all(self) -> None:
        """Create tables directly (tests / quick start). Production uses `alembic upgrade head`."""
        m.Base.metadata.create_all(self.engine)

    def migrate(self) -> None:
        """Apply Alembic migrations (`upgrade head`) to this database."""
        from pathlib import Path

        from alembic import command
        from alembic.config import Config

        if self.url.startswith("sqlite:///") and self.url != "sqlite:///:memory:":
            Path(self.url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        cfg = Config("alembic.ini")
        cfg.set_main_option("script_location", "migrations")
        cfg.cmd_opts = type("Opts", (), {"x": [f"url={self.url}"]})()
        command.upgrade(cfg, "head")

    def drop_all(self) -> None:
        m.Base.metadata.drop_all(self.engine)

    def _session(self) -> Session:
        return self._Session()

    def _embed_one(self, text: str) -> bytes:
        return _blob(self.embedder.encode([text])[0])

    # ---- writes ----
    def create_run(self, run_id: str, text: str, llm_backend: str = "unknown", created_at: datetime | None = None) -> None:
        """Insert a RUNNING run and its question embedding (idempotent for an existing run id)."""
        with self._session() as s, s.begin():
            if s.get(m.RunRow, run_id) is None:
                s.add(m.RunRow(run_id=run_id, request=text, status="RUNNING", llm_backend=llm_backend,
                               created_at=created_at or datetime.now(timezone.utc)))
                s.flush()  # Postgres enforces the FK at flush time; SQLite does not, so order the inserts explicitly
                s.add(m.QuestionRow(run_id=run_id, text=text, embedding=self._embed_one(text)))

    def add_sources(self, run_id: str, sources: list[Source]) -> None:
        """Upsert sources (deduplicated across runs by id; a longer text replaces a shorter one) and link them to the run."""
        with self._session() as s, s.begin():
            for src in sources:
                row = s.get(m.SourceRow, src.source_id)
                if row is None:
                    s.add(m.SourceRow(source_id=src.source_id, title=src.title, url=src.url, source_type=src.source_type,
                                      published_at=src.published_at, retrieved_at=src.retrieved_at, content=src.content, tool=src.tool))
                elif len(src.content) > len(row.content):
                    row.content = src.content
                if s.get(m.RunSourceRow, (run_id, src.source_id)) is None:
                    s.flush()
                    s.add(m.RunSourceRow(run_id=run_id, source_id=src.source_id))

    def add_claims(self, run_id: str, claims: list[dict]) -> None:
        """Upsert claims (and their evidence rows) for a run."""
        if not claims:
            return
        vecs = self.embedder.encode([c["claim"] for c in claims])
        with self._session() as s, s.begin():
            for c, v in zip(claims, vecs, strict=True):
                row = s.get(m.ClaimRow, (run_id, c["claim_id"]))
                fields = dict(claim=c["claim"], source_id=c["source_id"], evidence_quote=c["evidence_quote"],
                              extractor_confidence=float(c["extractor_confidence"]), verdict=c.get("verdict", "UNVERIFIED"),
                              verdict_rationale=c.get("verdict_rationale"), embedding=_blob(v))
                if row is None:
                    s.add(m.ClaimRow(run_id=run_id, claim_id=c["claim_id"], **fields))
                else:
                    for k, val in fields.items():
                        setattr(row, k, val)
                src = s.get(m.SourceRow, c["source_id"])
                span = find_quote(c["evidence_quote"], src.content) if src else None
                ev = s.get(m.EvidenceRow, (run_id, c["claim_id"]))
                vals = dict(source_id=c["source_id"], quote=c["evidence_quote"], quote_start=span[0] if span else None,
                            quote_end=span[1] if span else None, verified=span is not None)
                if ev is None:
                    s.add(m.EvidenceRow(run_id=run_id, claim_id=c["claim_id"], **vals))
                else:
                    for k, val in vals.items():
                        setattr(ev, k, val)

    def set_status(self, run_id: str, status: str) -> None:
        """Update a run's status without finishing it (e.g. AWAITING_APPROVAL)."""
        with self._session() as s, s.begin():
            row = s.get(m.RunRow, run_id)
            if row is not None:
                row.status = status

    def finish_run(self, run_id: str, status: str, *, workflow: str | None = None, budget: dict | None = None,
                   router: dict | None = None, finished_at: datetime | None = None, report: dict | None = None) -> None:
        """Mark the run finished (any terminal status incl. PARTIAL) and store its report."""
        with self._session() as s, s.begin():
            row = s.get(m.RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.status, row.workflow, row.budget, row.router = status, workflow, budget, router
            row.finished_at = finished_at or datetime.now(timezone.utc)
            if report is not None:
                text = str(report.get("executive_summary") or report.get("summary") or report.get("message") or row.request)
                existing = s.get(m.ReportRow, run_id)
                emb = self._embed_one(text)
                if existing is None:
                    s.add(m.ReportRow(run_id=run_id, content=report, summary=text, embedding=emb))
                else:
                    existing.content, existing.summary, existing.embedding = report, text, emb

    # ---- reads ----
    @staticmethod
    def _run_dict(r: m.RunRow) -> dict:
        return {"run_id": r.run_id, "request": r.request, "status": r.status, "workflow": r.workflow, "llm_backend": r.llm_backend,
                "created_at": _utc(r.created_at), "finished_at": _utc(r.finished_at), "budget": r.budget, "router": r.router}

    def get_run(self, run_id: str) -> dict | None:
        with self._session() as s:
            r = s.get(m.RunRow, run_id)
            return self._run_dict(r) if r else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        """Runs, newest first."""
        with self._session() as s:
            rows = s.scalars(select(m.RunRow).order_by(m.RunRow.created_at.desc()).limit(limit)).all()
            return [self._run_dict(r) for r in rows]

    def get_claims(self, run_id: str) -> list[dict]:
        with self._session() as s:
            rows = s.scalars(select(m.ClaimRow).where(m.ClaimRow.run_id == run_id).order_by(m.ClaimRow.claim_id)).all()
            return [self._claim_dict(c) for c in rows]

    @staticmethod
    def _claim_dict(c: m.ClaimRow) -> dict:
        return {"claim_id": c.claim_id, "claim": c.claim, "source_id": c.source_id, "evidence_quote": c.evidence_quote,
                "extractor_confidence": c.extractor_confidence, "verdict": c.verdict, "verdict_rationale": c.verdict_rationale}

    def get_sources(self, run_id: str) -> list[Source]:
        with self._session() as s:
            rows = s.scalars(select(m.SourceRow).join(m.RunSourceRow, m.RunSourceRow.source_id == m.SourceRow.source_id)
                             .where(m.RunSourceRow.run_id == run_id).order_by(m.SourceRow.source_id)).all()
            return [Source(source_id=r.source_id, title=r.title, url=r.url, source_type=r.source_type, published_at=_utc(r.published_at),
                           retrieved_at=_utc(r.retrieved_at), content=r.content, tool=r.tool) for r in rows]

    def get_report(self, run_id: str) -> dict | None:
        with self._session() as s:
            r = s.get(m.ReportRow, run_id)
            return r.content if r else None

    def count_sources(self) -> int:
        with self._session() as s:
            return len(s.scalars(select(m.SourceRow.source_id)).all())

    def delete_run(self, run_id: str) -> None:
        """Remove a run and its dependent rows."""
        with self._session() as s, s.begin():
            for model in (m.EvidenceRow, m.ClaimRow, m.ReportRow, m.RunSourceRow, m.QuestionRow):
                s.execute(delete(model).where(model.run_id == run_id))
            s.execute(delete(m.RunRow).where(m.RunRow.run_id == run_id))

    # ---- similarity search (brute-force cosine; swap for pgvector behind this interface later) ----
    @staticmethod
    def _rank(q: np.ndarray, rows: list, min_similarity: float) -> list[tuple[float, Any]]:
        if not rows:
            return []
        mat = np.vstack([_vec(r.embedding) for r in rows])
        sims = mat @ q
        return sorted(((float(sim), r) for sim, r in zip(sims, rows, strict=True) if sim >= min_similarity), key=lambda t: -t[0])

    def search(self, query: str, kinds: tuple[str, ...] = ("questions", "claims", "reports"), k: int = 5,
               min_similarity: float = 0.0) -> list[dict]:
        """Top-k matches across past questions, claims and report summaries."""
        q = self.embedder.encode([query])[0]
        out: list[dict] = []
        with self._session() as s:
            if "questions" in kinds:
                rows = s.scalars(select(m.QuestionRow)).all()
                for sim, r in self._rank(q, rows, min_similarity):
                    rep = s.get(m.ReportRow, r.run_id)
                    out.append({"kind": "questions", "run_id": r.run_id, "text": r.text, "similarity": sim,
                                "report": rep.content if rep else None})
            if "claims" in kinds:
                rows = s.scalars(select(m.ClaimRow)).all()
                for sim, r in self._rank(q, rows, min_similarity):
                    out.append({"kind": "claims", "run_id": r.run_id, "text": r.claim, "similarity": sim, "claim": self._claim_dict(r)})
            if "reports" in kinds:
                rows = s.scalars(select(m.ReportRow)).all()
                for sim, r in self._rank(q, rows, min_similarity):
                    out.append({"kind": "reports", "run_id": r.run_id, "text": r.summary, "similarity": sim, "report": r.content})
        return sorted(out, key=lambda h: -h["similarity"])[:k]

    def prior_findings(self, query: str, *, now: datetime | None = None, max_age_days: int = 30, min_similarity: float = 0.55,
                       k: int = 8) -> list[dict]:
        """Previously SUPPORTED claims relevant to `query`, younger than max_age_days, whose source AND verbatim quote are still stored."""
        now = _utc(now) or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)
        q = self.embedder.encode([query])[0]
        out = []
        with self._session() as s:
            rows = s.scalars(select(m.ClaimRow).where(m.ClaimRow.verdict == "SUPPORTED")).all()
            for sim, c in self._rank(q, rows, min_similarity):
                run = s.get(m.RunRow, c.run_id)
                created = _utc(run.created_at)
                if created < cutoff:
                    continue
                src = s.get(m.SourceRow, c.source_id)
                if src is None or find_quote(c.evidence_quote, src.content) is None:
                    continue  # reuse requires the source and the verbatim quote to still be present
                out.append({"claim": self._claim_dict(c), "run_id": c.run_id, "similarity": sim, "age_days": (now - created).days,
                            "source": {"source_id": src.source_id, "url": src.url, "title": src.title, "source_type": src.source_type}})
        return out[:k]

    def diff_with_prior(self, request: str, new_claims: list[dict], *, now: datetime | None = None, similarity: float = 0.85,
                        exclude_run_id: str | None = None, min_question_similarity: float = 0.6) -> dict:
        """Compare this run's claims with the most similar earlier run's SUPPORTED claims: which are new vs unchanged."""
        q = self.embedder.encode([request])[0]
        prior_run = None
        with self._session() as s:
            rows = [r for r in s.scalars(select(m.QuestionRow)).all() if r.run_id != exclude_run_id]
            for _sim, r in self._rank(q, rows, min_question_similarity):
                run = s.get(m.RunRow, r.run_id)
                if run.status in ("SUCCESS", "PARTIAL"):
                    prior_run = r.run_id
                    break
            prior = ([c for c in s.scalars(select(m.ClaimRow).where(m.ClaimRow.run_id == prior_run, m.ClaimRow.verdict == "SUPPORTED")).all()]
                     if prior_run else [])
        if not prior:
            return {"prior_run_id": prior_run, "new": list(new_claims), "unchanged": []}
        pm = np.vstack([_vec(c.embedding) for c in prior])
        new, unchanged = [], []
        for c, v in zip(new_claims, self.embedder.encode([c["claim"] for c in new_claims]), strict=True):
            (unchanged if float((pm @ v).max()) >= similarity else new).append(c)
        return {"prior_run_id": prior_run, "new": new, "unchanged": unchanged}
