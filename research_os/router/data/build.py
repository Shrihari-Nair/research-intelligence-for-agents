"""Merge template + LLM rows, clean, dedupe, balance, split (research domain)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from research_os.router.data.research_data import generate_templates
from research_os.router.data.split import split
from research_os.router.data.validate import normalize, validate

# Deterministic label-guide checks for the two intents with explicit lexical requirements.
CUES = {
    "LITERATURE_REVIEW": re.compile(
        r"\b(papers?|stud(?:y|ies)|research|literature|surveys?|review|academic|arxiv|published|publications?|"
        r"researchers?|state of the art|sota|seminal|citations?|reading list|works?)\b", re.I),
    "RESEARCH_MONITORING": re.compile(
        r"\b(monitor\w*|track\w*|watch\w*|keep me|keep an eye|alert\w*|notify|follow\w*|updates?|updated|subscribe\w*|"
        r"let me know|going forward|ongoing|regularly|weekly|daily|monthly|continuously|posted)\b", re.I),
}


def _tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def dedupe_near(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop rows whose token-set Jaccard with an earlier kept row is >= threshold (any intent). Keeps order."""
    kept_idx, kept_tok = [], []
    for idx, text in zip(df.index, df["text"], strict=True):
        tok = _tokens(text)
        if any(_jaccard(tok, k) >= threshold for k in kept_tok):
            continue
        kept_idx.append(idx)
        kept_tok.append(tok)
    return df.loc[kept_idx]


def clean_llm_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop LLM rows that violate the labeling guide's lexical cues; returns (clean, dropped per intent)."""
    keep = []
    dropped: dict[str, int] = {}
    for text, intent in zip(df["text"], df["intent"], strict=True):
        cue = CUES.get(intent)
        ok = cue is None or bool(cue.search(text))
        keep.append(ok)
        if not ok:
            dropped[intent] = dropped.get(intent, 0) + 1
    return df[keep].reset_index(drop=True), dropped


def _read_hard(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)[["text", "intent"]] if Path(path).exists() else pd.DataFrame(
        columns=["text", "intent"])


def build_dataset(cfg: dict, intents: list[str]) -> dict:
    """Return {"all","train","val","test","stats"} DataFrames/dict. Pure function of cfg + files on disk."""
    d, seed = cfg["data"], cfg["seed"]
    thr = d["near_duplicate_jaccard"]
    tpl = generate_templates(seed, d["template_per_intent"]).assign(source="template")
    llm_path = Path(d["llm_examples_dir"]) / "llm_generated.jsonl"
    llm_raw = pd.read_json(llm_path, lines=True)[["text", "intent"]] if llm_path.exists() else pd.DataFrame(
        columns=["text", "intent"])
    llm_raw = llm_raw[llm_raw["intent"].isin(intents)]
    llm, dropped_cues = clean_llm_rows(llm_raw)
    llm = llm.assign(source="llm")

    combined = pd.concat([llm, tpl], ignore_index=True)  # LLM rows first: they win near-duplicate ties
    hard = _read_hard(d["hard_test"])
    hard_tok = [_tokens(t) for t in hard["text"]]
    not_hard = [not any(_jaccard(_tokens(t), h) >= thr for h in hard_tok) for t in combined["text"]]
    n_hard_removed = int((~pd.Series(not_hard)).sum())
    combined = combined[not_hard].reset_index(drop=True)
    before = len(combined)
    combined = dedupe_near(combined, thr).reset_index(drop=True)

    avail = combined["intent"].value_counts()
    cap = int(min(d["max_per_intent"], avail.reindex(intents).min()))
    parts = [g.sample(n=cap, random_state=seed) for _, g in combined.groupby("intent")]
    all_df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    validate(all_df[["text", "intent"]], intents)
    tr, va, te = split(all_df[["text", "intent"]], seed, (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]))
    stats = {
        "seed": seed, "per_intent_cap": cap, "total": len(all_df), "near_duplicates_removed": before - len(combined),
        "rows_removed_for_hard_test_overlap": n_hard_removed, "llm_rows_dropped_by_label_cues": dropped_cues,
        "per_intent": {i: {"llm": int(((all_df.intent == i) & (all_df.source == "llm")).sum()),
                           "template": int(((all_df.intent == i) & (all_df.source == "template")).sum())}
                       for i in intents},
        "split_sizes": {"train": len(tr), "validation": len(va), "test": len(te), "hard_test": len(hard)},
    }
    return {"all": all_df, "train": tr, "val": va, "test": te, "stats": stats}
