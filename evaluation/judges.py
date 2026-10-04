"""LLM-as-judge for evaluation (role `eval_judge`, a different model from the pipeline's verifier). Cached, temperature 0.

Always report `agreement` against the hand-labelled gold claims next to any judged metric.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from research_os.agents.text import ShortIds
from research_os.llm.models import model_for_role
from research_os.llm.structured import complete_structured
from research_os.prompts import load_prompt


class _Label(BaseModel):
    id: str
    label: Literal["SUPPORTED", "UNSUPPORTED"]


class _Labels(BaseModel):
    labels: list[_Label]


class _Rating(BaseModel):
    id: str
    score: int = Field(ge=0, le=2)


class _Ratings(BaseModel):
    ratings: list[_Rating]


def _cover(want: set[str], got: list[str]) -> None:
    if set(got) - want:
        raise ValueError(f"unknown ids {sorted(set(got) - want)}")
    if want - set(got):
        raise ValueError(f"missing ids {sorted(want - set(got))}")


class Judge:
    """Batch judge. `llm` is usually a CachedLLM so reruns are free and identical."""

    def __init__(self, llm, batch_size: int = 10, model: str | None = None):
        self.llm, self.batch_size = llm, batch_size
        self.model = model or model_for_role("eval_judge")
        self._support, self._relevance = load_prompt("judge_claim_support"), load_prompt("judge_source_relevance")

    def claim_support(self, items: list[dict]) -> dict[str, str]:
        """items: {id, claim, quote, source}. Returns id -> SUPPORTED | UNSUPPORTED. The model sees short ids (k1..kN)."""
        out: dict[str, str] = {}
        for i in range(0, len(items), self.batch_size):
            batch = items[i: i + self.batch_size]
            ids = ShortIds([x["id"] for x in batch], "k")
            block = "\n\n".join(f"<<<ITEM id={ids.short(x['id'])}\nClaim: {x['claim']}\nPassage: {x['quote']}\nSource: {x['source']}\nITEM>>>" for x in batch)
            want = {ids.short(x["id"]).lower() for x in batch}
            parsed, _ = complete_structured(self.llm, self._support.render(items=block), _Labels, model=self.model, max_tokens=2048,
                                            meta={"prompt": self._support.id},
                                            validate=lambda d, want=want: _cover(want, [x.id.strip().lower() for x in d.labels]))
            out.update({ids.real(x.id): x.label for x in parsed.labels})
        return out

    def source_relevance(self, question: str, sources: list[dict]) -> dict[str, int]:
        """sources: {id, title, snippet}. Returns id -> 0|1|2. The model sees short ids (r1..rN)."""
        out: dict[str, int] = {}
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i: i + self.batch_size]
            ids = ShortIds([x["id"] for x in batch], "r")
            block = "\n\n".join(f"<<<SOURCE id={ids.short(x['id'])}\nTitle: {x['title']}\nText: {x['snippet'][:600]}\nSOURCE>>>" for x in batch)
            want = {ids.short(x["id"]).lower() for x in batch}
            parsed, _ = complete_structured(self.llm, self._relevance.render(question=question, items=block), _Ratings, model=self.model,
                                            max_tokens=1024, meta={"prompt": self._relevance.id},
                                            validate=lambda d, want=want: _cover(want, [x.id.strip().lower() for x in d.ratings]))
            out.update({ids.real(x.id): x.score for x in parsed.ratings})
        return out


def agreement(judged: dict[str, str], gold: dict[str, str]) -> dict:
    """Accuracy and Cohen's kappa between judge labels and gold labels over the shared ids."""
    ids = sorted(set(judged) & set(gold))
    n = len(ids)
    if n == 0:
        return {"n": 0, "accuracy": None, "kappa": None}
    acc = sum(judged[i] == gold[i] for i in ids) / n
    classes = sorted({judged[i] for i in ids} | {gold[i] for i in ids})
    pe = sum((sum(judged[i] == c for i in ids) / n) * (sum(gold[i] == c for i in ids) / n) for c in classes)
    kappa = None if pe == 1.0 else (acc - pe) / (1 - pe)
    return {"n": n, "accuracy": acc, "kappa": kappa}
