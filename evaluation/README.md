# Evaluation

Compares four systems on 120 research tasks, plus three harness ablations, from replayed data.

| System | Routing | Harness |
|---|---|---|
| A | LLM decides the workflow for every request | minimal |
| B | MiniLM classifier always decides | minimal |
| C | Hybrid: classifier unless unsure, then LLM | minimal |
| D | Hybrid | full: budgets, verifier gate, recovery |
| D-no-verifier / D-no-recovery / D-no-budgets | as D, minus one component | |

"Minimal" = no verifier (claims pass straight to synthesis, nothing is gated), no recovery, only a global step cap. Entity extraction is
deterministic code shared by all systems.

## Reproduce

```bash
make eval            # replay: frozen sources + LLM cache; writes experiments/results/{summary.json,summary.md,scores.json,runs/}
python -m evaluation.runner --offline --force   # proves reproducibility: every LLM call must be a cache hit
make eval-faults     # failure injection (offline)
make eval-record     # LIVE recording of the corpus (costs API calls; web content drifts)
```

* **Frozen corpus** (`cassettes/<task>.sources.json`): every source seen when each task was run once live (Tavily, GitHub, arXiv, PubMed, fetch).
  In replay, any system's tool calls are served from the task's pool (token-overlap search), so systems that issue different queries are compared
  on the same evidence. `cassettes/<task>.jsonl` is the exact call recording of that live run.
* **LLM cache** (`cassettes/llm_cache.jsonl`, `judge_cache.jsonl`): content-addressed (model, mode, prompt). A cache miss calls the LLM and stores
  the answer, so the first run costs API calls and every later run is identical and free.
* Deterministic fields are listed in `runner.DETERMINISTIC_KEYS` (test: a second offline run matches the first on every one). Wall-clock fields are not.

## Metrics

Router (workflow accuracy, macro F1, routing latency), research quality (task completion, end-to-end success, unsupported-claim rate,
citation correctness, source relevance), agent system (tool selection, plan success, recovery success), efficiency (LLM calls, tokens, cost,
tool calls, latency). Citation correctness = quote verbatim in the cited source **and** judged to support the claim.

## Honest caveats

* Tasks, labels and 60 gold claim labels were written by the project author (gold claims were labelled **before** the judge ran).
* The judge (`eval_judge`, `configs/models.yaml`) is an LLM; its agreement with the gold labels is reported with every judged metric. Judge errors
  in either direction move the numbers.
* One live recording and one replay per system: no run-to-run variance. Confidence intervals are bootstrap over *tasks*, not over repeated runs.
* Replay is a snapshot of the web at recording time (see `cassettes/RECORDING.json`).
* The recording used the pipeline as it was then: after it, prompts were changed to short ids (a fix for judge id mis-copies). Evaluation replays
  re-run the current pipeline against the frozen sources; only the LLM answers for changed prompts were regenerated live.
* **Citation correctness is not independent of the unsupported-claim rate.** Evidence quotes are verbatim-checked upstream, so
  `citation_correctness = 1 - unsupported_rate` on every system (verified in `scores.json`). Read them as one judged quantity, not two.
* Numbers inside free-text report summaries/findings are not independently checked against the quotes (only claim ids are validated).
* Recovery metrics include the first execution's failures and the replan decision (carried over by `merge_history`).
