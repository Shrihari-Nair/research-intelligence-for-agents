"""Sample claims from system D's runs on the gold tasks for HAND labelling (judge-agreement check).

Run BEFORE the judge has seen these claims, so labels cannot be influenced by its answers.
Usage: python -m evaluation.export_gold_candidates [--n 60] [--per-task 3] [--seed 42] > candidates.txt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluation.validate_dataset import load_tasks
from research_os.common import load_config


def sample_candidates(cfg: dict, n: int, per_task: int, seed: int, system: str = "D") -> list[dict]:
    """Up to `per_task` verbatim-quoted claims per gold task, `n` in total, shuffled with a fixed seed."""
    rng = random.Random(seed)
    gold = [t for t in load_tasks(cfg["paths"]["tasks"]) if t.gold]
    pool: list[dict] = []
    for t in gold:
        p = Path(cfg["paths"]["runs"]) / system / f"{t.id}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        items = [{"task_id": t.id, "question": t.input, "claim_id": e["claim_id"], "claim": e["claim"], "quote": e["quote"],
                  "source_title": e["source_title"], "source_type": e["source_type"], "verbatim": e["verbatim"]} for e in rec["evidence_checks"]]
        rng.shuffle(items)
        pool += items[:per_task]
    rng.shuffle(pool)
    return pool[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cands = sample_candidates(load_config("configs/eval.yaml"), args.n, args.per_task, args.seed)
    if args.json:
        print(json.dumps(cands))
        return
    for i, c in enumerate(cands, 1):
        print(f"[{i:02d}] {c['task_id']} | Q: {c['question']}\n     CLAIM: {c['claim']}\n     QUOTE: {c['quote']}\n     ({c['source_type']}) {c['source_title'][:70]}\n")


if __name__ == "__main__":
    main()
