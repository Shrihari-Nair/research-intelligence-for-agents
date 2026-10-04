"""One-off LLM generation of labelled research utterances (saved to data/examples for offline reproducibility)."""
from __future__ import annotations

import argparse
import logging
from datetime import date

from pydantic import BaseModel, Field

from research_os.common import load_config, load_dotenv, load_intents, setup_logging
from research_os.llm.gemini_client import GeminiLLMClient
from research_os.llm.models import model_for_role, provider_settings
from research_os.llm.structured import complete_structured

log = logging.getLogger(__name__)
PROMPT_VERSION = "datagen.v1"

DEFINITIONS = {
    "FACT_LOOKUP": "asks for ONE concrete checkable fact (a date, version, number, name, license, yes/no)",
    "TECHNOLOGY_COMPARISON": "asks to compare two or more NAMED technologies/frameworks/libraries on technical dimensions",
    "LITERATURE_REVIEW": "asks for papers, surveys, state of the art, or what the research says on a topic",
    "TECHNICAL_DEEP_DIVE": "asks how ONE technology, system or algorithm works internally",
    "COMPETITIVE_INTELLIGENCE": "asks about companies/commercial products, their competitors, positioning, pricing or strategy",
    "MARKET_RESEARCH": "asks about market size, growth, trends, adoption rates, pricing levels or forecasts, wanting figures",
    "RESEARCH_MONITORING": "asks for ONGOING tracking: monitor, watch, track, keep me updated, alert me, notify me going forward",
    "GENERAL_RESEARCH": "open-ended research: 'tell me about X', 'what is X', 'overview of X', or a ONE-OFF 'latest on X'",
}
STYLES = ["typed (clean sentences, normal punctuation)",
          "spoken (speech-to-text style: lowercase, no punctuation, fillers like um/so/uh, some run-ons)",
          "terse or indirect (very short, or a roundabout phrasing, but still unmistakably this intent)"]


class GeneratedBatch(BaseModel):
    """Schema the model must return."""

    utterances: list[str] = Field(min_length=1)


def build_prompt(intent: str, style: str, n: int) -> str:
    """Prompt for one (intent, style) batch."""
    others = ", ".join(i for i in DEFINITIONS if i != intent)
    return (
        f"You are generating training data for an intent classifier for a research assistant.\n"
        f"Intent: {intent} - the user {DEFINITIONS[intent]}.\n"
        f"Write {n} DIFFERENT user requests with this intent. Style: {style}.\n"
        "Vary length (3 to 30 words), topic (software, ML, science, business) and the named technologies/companies.\n"
        f"They must NOT be better described by these other intents: {others}.\n"
        'Return ONLY JSON: {"utterances": ["...", "..."]}'
    )


def generate_rows(llm, intent: str, n_per_style: int, model: str) -> list[dict]:
    """Generate labelled rows for one intent across all styles (schema-validated, max 2 retries per call)."""
    rows: list[dict] = []
    for style in STYLES:
        out, _ = complete_structured(llm, build_prompt(intent, style, n_per_style), GeneratedBatch, model=model,
                                     max_tokens=2048)
        for text in out.utterances:
            text = text.strip()
            if len(text.split()) >= 3:
                rows.append({"text": text, "intent": intent, "style": style, "model": model,
                             "prompt_version": PROMPT_VERSION, "generated_on": date.today().isoformat()})
    return rows


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--n-per-style", type=int, default=25)
    args = ap.parse_args()
    setup_logging()
    load_dotenv()
    cfg = load_config(args.config)
    prov, model = provider_settings(), model_for_role("data_generation")
    llm = GeminiLLMClient(model, prov["api_key_env"], prov["timeout_s"] * 3)
    rows: list[dict] = []
    for intent in load_intents(cfg["paths"]["intents"]):
        got = generate_rows(llm, intent, args.n_per_style, model)
        log.info("%s: %d rows", intent, len(got))
        rows += got
    out = f"{cfg['data']['llm_examples_dir']}/llm_generated.jsonl"
    pd.DataFrame(rows).to_json(out, orient="records", lines=True)
    log.info("wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
