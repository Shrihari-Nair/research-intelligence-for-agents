# Evaluation tasks

`tasks.jsonl`: 120 research tasks (15 per intent): 72 normal, 24 ambiguous, 24 "tempting" (designed to invite unsupported claims:
superlatives, precise figures, forecasts, universals). Labels (`expected_intent`, `required_source_types`, `min_supported_claims`)
were written by the project author following `data/LABELING_GUIDE.md`. `gold: true` marks the 32-task hand-labelled subset used to
check the LLM judge (M20); claim-level support labels for sampled claims live in `gold_claims.jsonl` (added after recording).

The tasks never appear in the router's train/validation/test/hard_test data (checked by `python -m evaluation.validate_dataset`).
Cassettes (`../cassettes/`) are recorded once with live tools and a real LLM; evaluation replays them (see `evaluation/README`).
