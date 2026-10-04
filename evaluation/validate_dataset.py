"""Validate evaluation/dataset/tasks.jsonl: schema, balance, uniqueness, workflow consistency, router-data independence, cassettes."""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from research_os.common import load_intents, setup_logging
from research_os.router.data.validate import normalize

log = logging.getLogger(__name__)
TASKS = "evaluation/dataset/tasks.jsonl"
CASSETTES = "evaluation/cassettes"


class Task(BaseModel):
    """One evaluation task."""

    id: str
    input: str = Field(min_length=3)
    expected_intent: str
    expected_workflow: str
    required_source_types: list[str] = Field(default_factory=list)
    must_have_citations: bool = True
    min_supported_claims: int = Field(ge=1)
    kind: Literal["normal", "ambiguous", "tempting"]
    gold: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Task:
        if self.expected_intent not in load_intents():
            raise ValueError(f"unknown intent {self.expected_intent!r}")
        if self.expected_workflow != self.expected_intent.lower():
            raise ValueError(f"workflow {self.expected_workflow!r} does not match intent {self.expected_intent!r}")
        return self


def load_tasks(path: str | Path = TASKS) -> list[Task]:
    """Parse and validate every task."""
    return [Task(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def _tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def near_duplicate_of(text: str, others: list[str], threshold: float) -> bool:
    """True if `text` has token-set Jaccard >= threshold with any of `others`."""
    t = _tokens(text)
    return any(len(t & o) / max(len(t | o), 1) >= threshold for o in map(_tokens, others))


def validate_dataset(path: str | Path = TASKS, router_data_dir: str = "data", *, require_cassettes: bool = True,
                     cassette_dir: str = CASSETTES, balance_tol: int = 2, fail_on_overlap: bool = True,
                     overlap_jaccard: float = 0.8) -> dict:
    """Return a report dict or raise ValueError describing the first class of problem found."""
    tasks = load_tasks(path)
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    norm = [normalize(t.input) for t in tasks]
    if len(norm) != len(set(norm)):
        raise ValueError("duplicate task inputs")
    per = Counter(t.expected_intent for t in tasks)
    if max(per.values()) - min(per.get(i, 0) for i in load_intents()) > balance_tol:
        raise ValueError(f"intent imbalance: {dict(per)}")
    router_texts: list[str] = []
    for split in ("train", "validation", "test", "hard_test"):
        f = Path(router_data_dir) / f"{split}.jsonl"
        if f.exists():
            router_texts += [json.loads(line)["text"] for line in f.read_text().splitlines() if line.strip()]
    router_norm = {normalize(x) for x in router_texts}
    overlap = [t.id for t, n in zip(tasks, norm, strict=True) if n in router_norm or near_duplicate_of(t.input, router_texts, overlap_jaccard)]
    if overlap and fail_on_overlap:
        raise ValueError(f"overlap with router training/eval data: {overlap}")
    missing = [t.id for t in tasks if not ((Path(cassette_dir) / f"{t.id}.jsonl").exists() and (Path(cassette_dir) / f"{t.id}.sources.json").exists())]
    if missing and require_cassettes:
        raise ValueError(f"missing cassette(s) for {len(missing)} task(s): {missing[:10]}")
    return {"n": len(tasks), "per_intent": dict(per), "kinds": dict(Counter(t.kind for t in tasks)), "gold": sum(t.gold for t in tasks),
            "router_overlap": overlap, "missing_cassettes": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cassettes", action="store_true", help="do not require cassettes")
    args = ap.parse_args()
    setup_logging()
    rep = validate_dataset(require_cassettes=not args.no_cassettes)
    log.info("dataset OK: %s", {k: v for k, v in rep.items() if k != "missing_cassettes"})


if __name__ == "__main__":
    main()
