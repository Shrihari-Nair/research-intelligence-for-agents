# ROADMAP: CPU-Efficient Fine-Tuned Intent Router for Agentic Voice Systems

> Instructions for Claude Code: read this whole file before writing code. Work milestone by milestone, in order. Do not start a milestone until the previous one's acceptance criteria pass. Commit after each milestone.

---

## 0. Project summary

Build a small but complete ML + agentic-systems project:

- A **SetFit-fine-tuned MiniLM** classifier that routes a user utterance to one of 8 intents in a few milliseconds on **CPU**.
- A **LangGraph** workflow with 3 simple agents (booking, prescription, research) that the router dispatches to.
- A **confidence-based hybrid router**: high confidence routes directly, low confidence falls back to an LLM.
- A **reproducible benchmark** comparing three routing systems: LLM-only, MiniLM-only, Hybrid.
- A thin **FastAPI** `/chat` endpoint, with an optional voice layer at the end.

The MiniLM model does **not** replace the LLM. It answers one narrow question ("what does the user want?") quickly. The LLM and agents handle everything after that.

## 1. Hard rules

1. **Never fabricate numbers.** Every metric in the README or docs must come from a script run and be saved in `results/*.json`. Placeholders such as `TBD` are fine until real runs exist.
2. **Never touch the test set** during training, threshold selection, or hyperparameter tuning. Use validation for those.
3. **Everything is config-driven** (`configs/*.yaml`): model name, seed, intents, thresholds, paths. No magic constants in code.
4. **Fixed seeds** everywhere (default `42`). Report mean ± std over 3 seeds for the data-size ablation.
5. **CPU only.** No code path may require a GPU. Set `torch.set_num_threads` explicitly in benchmarks and log the value.
6. **Each script is runnable from the repo root** and prints a clear summary (see section 6).
7. **No real patient data.** All data is synthetic. State this limitation in the README.
8. **Emergency safety rule:** the `EMERGENCY` intent must never be silently dropped. If `EMERGENCY` has the top probability, or its probability exceeds `emergency_floor` (config, default `0.30`), route to the emergency handler regardless of the general confidence threshold. Optimize for **recall** on this class and report it separately.
9. Keep the agents deliberately simple. The ML router is the main contribution.

## 2. Tech stack

- Python 3.10+
- `torch` (CPU build), `transformers`, `sentence-transformers`, `setfit` (>=1.0), `datasets`
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `psutil`, `pyyaml`
- `langgraph`, `fastapi`, `uvicorn`, `pydantic`
- `anthropic` (LLM fallback; behind an interface, see M5)
- `pytest`, `ruff`

Default base model: `sentence-transformers/all-MiniLM-L6-v2` (configurable).

### SetFit API notes (checked against the huggingface/setfit README, quickstart and release notes)

- Use the v1.0+ API: `from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset`. The older `SetFitTrainer` / `DistillationSetFitTrainer` were replaced by `Trainer` / `DistillationTrainer`. Do **not** use `SetFitTrainer` and do not pass `num_epochs` to `trainer.train()`; set training options on `TrainingArguments`.
- `SetFitModel` combines a sentence-transformers body with a classification head. The default head is a scikit-learn logistic regression; a differentiable PyTorch head (`use_differentiable_head=True`) is optional. Keep the default.
- Pass labels via `SetFitModel.from_pretrained(name, labels=[...])` so predictions can return label strings.
- Few-shot subsets: `sample_dataset(dataset, label_column="label", num_samples=N)`. Use this in the ablation (N = 10, 25, 50, 75).
- Column names: the trainer expects `text` and `label`. If your data differs, pass `column_mapping={"sentence": "text", "label": "label"}` to `Trainer`.
- Evaluation: `trainer.evaluate(test_dataset)` accepts an explicit dataset.
- Saving/loading: `model.save_pretrained(path)` then `SetFitModel.from_pretrained(path)`. **Always save explicitly after training.** Do not rely on checkpoints written during training; an upstream issue (#526) reports that mid-training checkpoints can contain an untrained head.
- Evaluation-during-training argument name was renamed upstream from `evaluation_strategy` to `eval_strategy`. Check the installed version's `TrainingArguments` signature before setting it.
- The SetFit quickstart notes that better performance comes from more data rather than more training, so keep `num_epochs` and `num_iterations` small on CPU and tune on validation.
- Verify `predict_proba` and `predict` (including a `batch_size` argument) against the installed version with a 5-line check in M1. The router in M3 depends on `predict_proba` for confidence.

Minimal reference (confirm against the installed version in M1):

```python
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2", labels=INTENTS
)
args = TrainingArguments(batch_size=16, num_epochs=1, seed=42)  # tune on val
trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
metrics = trainer.evaluate(test_ds)   # only for final reporting
model.save_pretrained("models/setfit_router")
probs = model.predict_proba(["can I move my appointment to friday"])
```

## 3. Intents

| Intent | Example |
|---|---|
| `BOOK_APPOINTMENT` | "I want to see Dr Sharma tomorrow" |
| `RESCHEDULE_APPOINTMENT` | "Can I move my appointment to Friday?" |
| `CANCEL_APPOINTMENT` | "Cancel my appointment" |
| `PRESCRIPTION_QUERY` | "What medicines did the doctor give me?" |
| `FOLLOWUP_QUERY` | "When should I follow up?" |
| `MEDICAL_RESEARCH` | "What are the latest treatments for migraine?" |
| `GENERAL_QUERY` | "What services do you provide?" |
| `EMERGENCY` | "I'm having severe chest pain" |

Do not add more intents. Store the list in `configs/intents.yaml` and load it everywhere.

## 4. Target repository layout

```
agentic-intent-router/
├── configs/
│   ├── intents.yaml
│   ├── training.yaml
│   └── routing.yaml            # threshold, emergency_floor, llm model
├── data/
│   ├── raw/
│   ├── processed/              # train.csv, val.csv, test.csv, hard_test.csv
│   └── intents.csv
├── src/
│   ├── data/        (prepare.py, split.py, validate.py)
│   ├── training/    (train_setfit.py, baseline.py, ablation.py)
│   ├── models/      (router.py, calibration.py)
│   ├── llm/         (client.py, anthropic_client.py, mock_client.py)
│   ├── agents/      (booking.py, prescription.py, research.py, emergency.py, tools.py)
│   ├── graph/       (state.py, workflow.py)
│   ├── systems/     (llm_only.py, minilm_only.py, hybrid.py)
│   ├── api/         (app.py)
│   └── evaluation/  (evaluate.py, benchmark.py, compare_systems.py, e2e_tasks.py)
├── tests/
├── results/                    # JSON/PNG outputs, git-tracked
├── models/                     # saved fine-tuned models, gitignored
├── requirements.txt
├── README.md
└── demo.py
```

---

## 5. Milestones

### M0. Scaffold (about 30 min)

Tasks:
- Create the repo layout, `requirements.txt`, `.gitignore` (ignore `models/`, `.venv`, `__pycache__`), `ruff` config, and a `Makefile` with targets: `data`, `train`, `evaluate`, `benchmark`, `compare`, `test`.
- Write the three YAML configs with sensible defaults.
- Write a `src/common.py` with helpers: `load_config`, `set_seed`, `get_hardware_info()` (CPU model, cores, RAM, torch version, thread count).

Acceptance:
- `pip install -r requirements.txt` works on a clean venv.
- `pytest` runs (even with a single placeholder test).

---

### M1. Prove CPU fine-tuning works (about 1-2 hours)

Goal: confirm the SetFit toolchain runs end to end on this machine before building anything real.

Tasks:
- `src/training/smoke_sst2.py`: load SST-2 via `datasets`, sample 8 examples per class with `sample_dataset`, use a small slice (e.g. 200) for eval, fine-tune MiniLM with SetFit for 1 epoch, print accuracy and wall-clock training time. Mirror the SetFit README/quickstart example, adapted to MiniLM.
- In the same script, verify the API assumptions from the "SetFit API notes" section: `predict_proba` output shape, label handling, and a `save_pretrained` / `from_pretrained` round trip that gives identical predictions.

Acceptance:
- Script completes on CPU without errors and prints accuracy plus training time.
- Training time is recorded in `results/smoke_sst2.json`.

---

### M2. Dataset (about 2-3 hours)

Goal: ~600 clean, diverse, labeled utterances.

Tasks:
- Generate `data/intents.csv` with columns `text,intent`. Target 75 examples per intent (about 600 total). Generate synthetic data programmatically and/or with an LLM, then review it. Required diversity per intent:
  - Different phrasings, lengths (3 to 30 words), politeness levels, and fillers typical of speech ("um", "so", "can you").
  - Varied entities (doctor names, days, dates, drug names, conditions).
  - Some utterances without punctuation or capitalization (to mimic STT output).
  - A few that are indirect or colloquial.
- `src/data/validate.py`: check for duplicates (exact and near-duplicate via normalized text), label balance, empty strings, and unknown labels. Fail loudly.
- `src/data/split.py`: stratified 70/15/15 split into `train.csv`, `val.csv`, `test.csv` with a fixed seed. Assert **zero overlap** between splits.
- Build a separate **`hard_test.csv`** (about 40 to 60 rows) of ambiguous or multi-intent utterances, such as: "I have an appointment coming up but I'm not sure whether I should move it because my symptoms changed." Label each with the single best intent. This set is used in M5/M7 to show where the hybrid router helps. Keep it out of train/val.

Acceptance:
- `make data` produces all processed files; `validate.py` passes.
- A test asserts no train/val/test leakage.
- README section "Dataset" documents how data was made and states that it is synthetic.

---

### M3. Baseline and fine-tuning (about 2-4 hours)

Goal: a fair baseline, then a fine-tuned model that beats it.

Tasks:
- `src/training/baseline.py`: **baseline** = pretrained MiniLM (no contrastive fine-tuning) with a Logistic Regression head trained on frozen embeddings of `train.csv`. Evaluate on `test.csv`.
- `src/training/train_setfit.py`: fine-tune with SetFit (contrastive fine-tune of the sentence transformer, then classification head). Hyperparameters in `configs/training.yaml` (batch size, epochs, num_iterations, learning rate). Early iterations can use small `num_iterations` to keep CPU time low. Select hyperparameters using `val.csv` only. Use `Trainer` + `TrainingArguments` (not `SetFitTrainer`). Save explicitly with `model.save_pretrained("models/setfit_router")` after training; do not use mid-training checkpoints. Reload in a test and assert predictions match.
- `src/models/router.py`: class `IntentRouter` with:
  - `load(path)`
  - `predict(text) -> {"intent": str, "confidence": float, "probs": dict}` where confidence is the max of `predict_proba`
  - `predict_batch(texts)`
- `src/training/ablation.py`: train with **10, 25, 50, 75 examples per intent**, 3 seeds each; evaluate on the test set; save a table (mean ± std) to `results/ablation.json` and a plot to `results/ablation.png`.

Acceptance:
- Baseline and fine-tuned metrics both saved in `results/` (`baseline.json`, `finetuned.json`).
- Fine-tuned macro F1 is reported next to baseline macro F1 (whether or not it wins; report honestly).
- Ablation table exists with 4 sizes x 3 seeds.

---

### M4. Evaluation harness (about 2 hours)

Tasks:
- `src/evaluation/evaluate.py`: for a given model (baseline or fine-tuned), compute accuracy, macro precision/recall/F1, per-intent F1, confusion matrix (saved as PNG), and **EMERGENCY recall** on its own line.
- `src/evaluation/benchmark.py`: latency and resource benchmark:
  - Load model, run 50 warm-up requests, then **1,000 single-request inferences** (batch size 1), timing each with `time.perf_counter`.
  - Report P50, P95, P99, mean, throughput (req/s).
  - Report model size on disk (MB), process RSS memory (MB) after load via `psutil`, and training time (read from the training log).
  - Log hardware info and thread count into the JSON output.
- Both scripts print a banner in the format:

```
=============================
Intent Router Evaluation
=============================
Model: MiniLM
Fine-tuned: Yes
Accuracy:       XX.XX%
Macro F1:       XX.XX%
P50 latency:    XX ms
P95 latency:    XX ms
Model size:     XX MB
CPU memory:     XX MB
=============================
```

Acceptance:
- `python -m src.evaluation.evaluate --model models/setfit_router` and `python -m src.evaluation.benchmark --model models/setfit_router` both run and write JSON to `results/`.
- Baseline vs fine-tuned can be compared with the same scripts.

---

### M5. Confidence-based routing and LLM fallback (about 2-3 hours)

Tasks:
- **Calibration/threshold:** `src/models/calibration.py`. On `val.csv`, sweep thresholds from 0.30 to 0.95. For each, compute coverage (share of requests handled directly by MiniLM) and accuracy on that covered subset. Save `results/threshold_sweep.json` and a coverage-vs-accuracy plot. Choose the threshold from validation (document the rule used, e.g. smallest threshold with covered-accuracy >= 97%) and write it to `configs/routing.yaml`. Do **not** hardcode 0.80 unless validation supports it.
- **LLM client interface** (`src/llm/client.py`): `class LLMClient: classify_intent(text, intents) -> {"intent": str, "reason": str}`.
  - `anthropic_client.py`: real implementation using the `anthropic` SDK; reads `ANTHROPIC_API_KEY` from the environment; model name from `configs/routing.yaml`; prompts the model to return strict JSON with one of the allowed intent labels; validates the output and falls back to `GENERAL_QUERY` on parse failure.
  - `mock_client.py`: deterministic offline stub for tests (e.g. keyword rules) with configurable artificial delay.
- **Routing logic** (`src/models/router.py` or `src/systems/hybrid.py`):

```
probs = minilm.predict(text)
if top intent == EMERGENCY or probs[EMERGENCY] >= emergency_floor: route to emergency
elif confidence >= threshold: route directly to the agent for that intent
else: ask LLM -> route to the agent for the LLM's intent
```

- Return a routing record: `{intent, confidence, source: "minilm"|"llm", llm_called: bool, latency_ms}`.

Acceptance:
- Unit tests cover: high-confidence path (no LLM call), low-confidence path (LLM called, using the mock), emergency override, malformed LLM output.
- `results/threshold_sweep.json` exists and `configs/routing.yaml` threshold is justified by it.

---

### M6. Agents and LangGraph workflow (about 2-4 hours)

Tasks:
- `src/agents/tools.py`: simple, deterministic, **mock** tools backed by in-memory data or SQLite seeded with fake doctors, slots, patients, prescriptions:
  - Booking: `check_slots()`, `book_appointment()`, `reschedule_appointment()`, `cancel_appointment()`
  - Prescription: `get_prescription()`, `get_followup()`
  - Research: `search_pubmed()`, `search_web()` (stubbed with canned results by default; optional real PubMed E-utilities behind a config flag, off by default)
- Agents: `booking.py` (handles book/reschedule/cancel), `prescription.py` (prescription + follow-up), `research.py`, `emergency.py` (returns a fixed safety message directing the user to local emergency services; makes no tool calls and no diagnosis). `GENERAL_QUERY` is handled by a small FAQ responder.
- `src/graph/workflow.py`: LangGraph `StateGraph` with state `{text, routing, agent_name, tool_calls, response}`:

```
START -> intent_router -> (conditional edge by intent) -> agent node -> response -> END
```

  The `intent_router` node is pluggable so the same graph can run all three systems in M7.
- Agents may use the LLM (via `LLMClient`) only to phrase final responses, and must work with the mock client for tests.

Acceptance:
- `python demo.py "Can I move my appointment to Friday?"` prints intent, confidence, agent, tool calls, and response.
- Integration tests: at least one scenario per intent runs through the graph with the mock LLM and asserts the expected agent and tool call.

---

### M7. Compare three systems (about 2-3 hours)

Systems (in `src/systems/`):
- **A, LLM-only:** every utterance goes to the LLM for intent classification, then the agent.
- **B, MiniLM-only:** MiniLM intent goes straight to the agent (emergency rule still applies).
- **C, Hybrid:** the M5 logic.

Tasks:
- `src/evaluation/e2e_tasks.py`: define ~40 end-to-end scenarios, each with an utterance, expected intent, expected agent, and expected tool call. Include some from `hard_test.csv`. "Task success" = correct agent **and** correct tool call.
- `src/evaluation/compare_systems.py`: run all three systems on (a) `test.csv`, (b) `hard_test.csv`, and (c) the e2e scenarios. Output:

| Metric | LLM-only | MiniLM-only | Hybrid |
|---|---|---|---|
| Intent accuracy | | | |
| Macro F1 | | | |
| P50 latency | | | |
| P95 latency | | | |
| CPU usage | | | |
| Cost/request | | | |
| LLM calls | | | |
| End-to-end task success | | | |

- Cost/request: compute from actual token usage and the configured per-token prices (put prices in `configs/routing.yaml`; mark them as configurable, not authoritative). MiniLM cost = 0.
- If no `ANTHROPIC_API_KEY` is available, run only B and the hybrid with the mock fallback, and mark LLM-only fields as `"not_run"` in the JSON. **Do not simulate or invent LLM numbers.**
- Save to `results/compare_systems.json` and a rendered `results/compare_systems.md`.

Acceptance:
- `python -m src.evaluation.compare_systems` produces the table and JSON with all fields filled or explicitly `not_run`.

---

### M8. API and voice layer (about 2-3 hours)

Tasks:
- `src/api/app.py`: FastAPI app with `POST /chat`:

Request:
```json
{ "text": "Can I move my appointment to Friday?" }
```
Response:
```json
{
  "intent": "RESCHEDULE_APPOINTMENT",
  "confidence": 0.93,
  "source": "minilm",
  "agent": "booking_agent",
  "response": "...",
  "latency_ms": 14
}
```
  Also add `GET /health`. Load the model once at startup.
- Voice (optional, do last): add an STT/TTS adapter interface with a no-op/text implementation by default. LiveKit integration is a stretch goal and must live behind a separate optional extra (`requirements-voice.txt`) so the core project stays CPU/light and testable. Flow: mic -> STT -> router -> LangGraph -> tool -> response -> TTS.

Acceptance:
- `uvicorn src.api.app:app` starts; `curl` against `/chat` returns the JSON above; an API test with `TestClient` passes.

---

### M9. Polish and documentation (about 1-2 hours)

Tasks:
- README with: overview and architecture diagram (ASCII or Mermaid), setup, `make` commands, dataset description and synthetic-data caveat, results tables **pulled from `results/*.json`** (generate with a script, e.g. `scripts/render_results.py`), design decisions (why SetFit, why a threshold, why an emergency floor), limitations, and future work.
- Limitations to state explicitly: synthetic and small dataset, in-distribution test set may overstate real-world accuracy, mock tools, latency measured on one machine (include hardware info).
- Ensure `ruff` and `pytest` are clean.
- Resume-style project summary: leave numeric claims as placeholders (`XX`) unless filled from `results/`.

Acceptance:
- A fresh clone can run: `make data && make train && make evaluate && make benchmark && make compare`.

---

## 6. Script conventions

- CLI via `argparse`; all take `--config` and `--seed` where relevant.
- Every script writes machine-readable JSON to `results/` and prints a human-readable summary.
- Use `logging`, not bare `print`, except for the final summary banner.
- Type hints and short docstrings on public functions.

## 7. Testing checklist (minimum)

- Data: no duplicates, no leakage, balanced labels, valid labels only.
- Router: output schema, confidence in [0, 1], probabilities sum to ~1.
- Routing: high-confidence, low-confidence, emergency override, LLM failure fallback.
- Graph: one scenario per intent end to end with mock LLM.
- API: `/health` and `/chat` happy path plus empty-text validation error.

## 8. Definition of done

- [ ] All milestone acceptance criteria met
- [ ] Fine-tuned vs baseline metrics saved and reported honestly
- [ ] Data-size ablation (10/25/50/75 x 3 seeds) saved
- [ ] Threshold chosen from validation sweep
- [ ] Three-system comparison produced (or LLM-only explicitly `not_run`)
- [ ] EMERGENCY recall reported separately
- [ ] README results generated from `results/`, no invented numbers
- [ ] Tests and lint pass

## 9. Suggested order if time is short

M0 -> M1 -> M2 -> M3 -> M4 (this alone is a solid ML project), then M5 -> M6 -> M7 (the agentic and hybrid story), then M8 -> M9.
