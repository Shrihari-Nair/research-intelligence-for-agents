"""Verifier accuracy on the 30-pair hand-labelled gold set (evaluation/dataset/verifier_gold.jsonl).

Usage: python -m evaluation.verifier_eval [--live]   (--live uses the configured Gemini verifier_judge model; needs GOOGLE_API_KEY)
Writes experiments/results/verifier_eval.json. Labels were written by the project author, not independent annotators.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from research_os.agents.verifier import Verifier
from research_os.common import get_hardware_info, load_dotenv, setup_logging, write_json
from research_os.contracts import ResearchStep, Source, source_id_for_url
from research_os.harness.context import RunContext
from research_os.tools.registry import ToolRegistry

log = logging.getLogger(__name__)
VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")
GOLD = "evaluation/dataset/verifier_gold.jsonl"


def load_gold(path: str = GOLD) -> list[dict]:
    """The labelled pairs."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate_pairs(gold: list[dict], llm) -> dict:
    """Run the verifier (deterministic layer + judge) over every pair, each as its own one-claim run."""
    now = datetime.now(timezone.utc)
    ctx = RunContext.create("verifier-eval", {}, llm=llm, tools=ToolRegistry())
    claims = []
    for g in gold:
        url = f"https://gold.example/{g['id']}"
        ctx.sources.add(Source(source_id=source_id_for_url(url), title=g["id"], url=url, source_type=g["source_type"], published_at=now,
                               retrieved_at=now, content=g["source_text"], tool="gold"))
        claims.append({"claim_id": g["id"], "claim": g["claim"], "source_id": source_id_for_url(url), "evidence_quote": g["evidence_quote"],
                       "extractor_confidence": 1.0, "verdict": "UNVERIFIED", "verdict_rationale": None})
    out = Verifier().run(ResearchStep(id="verify", objective="v", agent="verifier", tools=[], depends_on=[],
                                      params={"workflow": "general_research"}), {"x": {"claims": claims}}, ctx, "gold")
    got = {c["claim_id"]: c for c in out["claims"]}
    confusion = {e: Counter() for e in VERDICTS}
    rows, correct = [], 0
    det_rejected = det_false = 0
    for g in gold:
        c = got[g["id"]]
        confusion[g["expected"]][c["verdict"]] += 1
        correct += c["verdict"] == g["expected"]
        det = c.get("verdict_source") == "deterministic"
        det_rejected += det
        det_false += det and g["expected"] == "SUPPORTED"
        rows.append({"id": g["id"], "kind": g["kind"], "expected": g["expected"], "got": c["verdict"], "deterministic_rejection": det})
    per_class = {e: sum(confusion[e].values()) and confusion[e][e] / sum(confusion[e].values()) for e in VERDICTS}
    return {"n": len(gold), "accuracy": correct / len(gold), "per_class_recall": per_class,
            "confusion": {e: dict(confusion[e]) for e in VERDICTS},
            "deterministic": {"rejected": det_rejected, "false_rejections_of_supported": det_false},
            "judge_calls": out["stats"]["judge_calls"], "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real Gemini verifier_judge model")
    args = ap.parse_args()
    setup_logging()
    load_dotenv()
    from research_os.llm.gemini_client import GeminiLLMClient
    from research_os.llm.models import model_for_role, provider_settings

    if not args.live:
        raise SystemExit("run with --live (a scripted judge is only meaningful inside tests)")
    prov = provider_settings()
    llm = GeminiLLMClient(model_for_role("verifier_judge"), prov["api_key_env"], prov["timeout_s"] * 3)
    res = evaluate_pairs(load_gold(), llm)
    res.update({"judge_model": model_for_role("verifier_judge"), "labels_by": "project author (hand-written, not independent)",
                "hardware": get_hardware_info()})
    write_json("experiments/results/verifier_eval.json", res)
    log.info("verifier accuracy %.3f per-class recall %s", res["accuracy"], res["per_class_recall"])


if __name__ == "__main__":
    main()
