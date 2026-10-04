# CPU-Efficient Intent Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SetFit-fine-tuned MiniLM intent router (8 intents, CPU, ms latency) with confidence-based LLM fallback, a LangGraph agent workflow, a 3-system benchmark, and a FastAPI `/chat` endpoint.

**Architecture:** `IntentRouter` (SetFit) yields `{intent, confidence, probs}`. A `HybridSystem` applies emergency-floor → threshold → LLM-fallback logic and returns a routing record. A LangGraph `StateGraph` takes a pluggable router node and dispatches to 4 agents + FAQ responder over mock tools. Evaluation scripts write JSON to `results/`; the README is rendered from those JSONs.

**Tech Stack:** Python 3.10+ (env has 3.12), torch (CPU), sentence-transformers, setfit>=1.0, datasets, scikit-learn, langgraph, fastapi, anthropic, pytest, ruff.

**Spec:** `ROADMAP.md` (repo root)

## Global Constraints

- Never fabricate numbers; every metric comes from a script run saved to `results/*.json`. `TBD`/`XX` placeholders OK until real runs.
- Never touch the test set for training, threshold selection, or hyperparameter tuning (use val).
- Everything config-driven (`configs/*.yaml`): model name, seed, intents, thresholds, paths, prices. No magic constants.
- Fixed seed `42`; ablation reports mean ± std over 3 seeds.
- CPU only; call `torch.set_num_threads` explicitly in benchmarks and log it.
- Every script runnable from repo root (`python -m src...`), argparse with `--config`/`--seed`, JSON to `results/`, `logging` (not `print`) except the final banner.
- All data synthetic; state this in README. Exactly the 8 intents, no more, loaded from `configs/intents.yaml`.
- EMERGENCY: if top intent is EMERGENCY or `probs[EMERGENCY] >= emergency_floor` (default 0.30) → emergency handler regardless of threshold. Report EMERGENCY recall separately.
- SetFit v1.0 API only: `SetFitModel, Trainer, TrainingArguments, sample_dataset`. No `SetFitTrainer`; no `num_epochs` in `trainer.train()`. Always `save_pretrained` explicitly (never rely on mid-training checkpoints, upstream #526). Check `eval_strategy` vs `evaluation_strategy` against the installed version.
- Do not simulate LLM numbers; without `ANTHROPIC_API_KEY`, LLM-only fields are `"not_run"`.
- Commit after each milestone (repo is not yet a git repo: Task 0 runs `git init`).

## Review Focus

1. Empty / whitespace-only / very long text to `IntentRouter.predict` and `/chat` — expect validation error (API 422) or graceful result, never a crash.
2. Emergency phrased with low overall confidence (e.g. "i cant breathe" where probs are spread) — expect emergency route via floor.
3. LLM returns prose, invalid JSON, or an intent outside the 8 — expect fallback to `GENERAL_QUERY`; LLM raises/timeouts — same fallback, flagged in record.
4. Agent asked to reschedule/cancel with no existing appointment, or a booking for a full slot — expect a clear response, not an exception.
5. STT-style input (lowercase, no punctuation, fillers) must classify like clean text; data and tests include such rows.

---

## File Structure

Follows ROADMAP §4 exactly. Plus: `src/common.py`, `src/data/generate.py` (synthetic data), `scripts/render_results.py`, `tests/conftest.py`. Each module has one responsibility; `systems/*` share the `RoutingSystem` protocol defined in Task 8.

## Open decisions (default chosen so work can proceed)

- **Data generation:** programmatic templates + entity slots (deterministic, offline) with hand-written extras for colloquial/indirect phrasing; optional LLM augmentation if `ANTHROPIC_API_KEY` set. Hard-test set is hand-written.
- **Package name:** repo root is the project (`src/` importable as a package via `src/__init__.py`), not nested `agentic-intent-router/`.

---

## M0 — Scaffold

### Task 1: Repo scaffold, configs, common helpers

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pyproject.toml` (ruff + pytest config), `Makefile`, `configs/intents.yaml`, `configs/training.yaml`, `configs/routing.yaml`, `src/__init__.py` (+ `__init__.py` in every `src/*` package), `src/common.py`, `tests/test_common.py`
- Directories: `data/raw data/processed results models scripts`

**Interfaces:**
- Produces: `load_config(path: str) -> dict`, `load_intents(path="configs/intents.yaml") -> list[str]`, `set_seed(seed: int) -> None`, `get_hardware_info() -> dict` (keys: `cpu_model, cores, ram_gb, torch_version, torch_threads`), `write_json(path, obj)`.

- [ ] **Step 1:** `git init`; write `.gitignore` (`models/ .venv __pycache__ .pytest_cache .ruff_cache`).
- [ ] **Step 2:** Create venv, write `requirements.txt` (pin loosely: `setfit>=1.0`, `sentence-transformers`, `transformers`, `datasets`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `psutil`, `pyyaml`, `langgraph`, `fastapi`, `uvicorn`, `pydantic`, `anthropic`, `httpx`, `pytest`, `ruff`; install torch via `--extra-index-url https://download.pytorch.org/whl/cpu`). Run `pip install -r requirements.txt` in a clean venv.
- [ ] **Step 3:** Write configs:

```yaml
# configs/intents.yaml
intents: [BOOK_APPOINTMENT, RESCHEDULE_APPOINTMENT, CANCEL_APPOINTMENT, PRESCRIPTION_QUERY, FOLLOWUP_QUERY, MEDICAL_RESEARCH, GENERAL_QUERY, EMERGENCY]
# configs/training.yaml
seed: 42
base_model: sentence-transformers/all-MiniLM-L6-v2
paths: {raw: data/intents.csv, processed: data/processed, model: models/setfit_router, results: results}
setfit: {batch_size: 16, num_epochs: 1, num_iterations: 10, learning_rate: 2.0e-5}
ablation: {sizes: [10, 25, 50, 75], seeds: [42, 43, 44]}
# configs/routing.yaml
threshold: null            # set by M5 calibration from val
emergency_floor: 0.30
target_covered_accuracy: 0.97
llm: {model: claude-haiku-4-5-20251001, max_tokens: 150, timeout_s: 10}
pricing_per_mtok: {input: 1.0, output: 5.0}   # configurable, NOT authoritative
```

- [ ] **Step 4: Failing test** `tests/test_common.py`: `load_intents()` returns 8 unique strings incl. `EMERGENCY`; `set_seed` makes `numpy`/`random`/`torch` reproducible; `get_hardware_info()` has required keys.
- [ ] **Step 5:** Run `pytest tests/test_common.py -v` → FAIL; implement `src/common.py`; rerun → PASS.
- [ ] **Step 6:** Makefile targets `data train evaluate benchmark compare test` (initially echo/stub; filled as milestones land). `ruff check .` clean.
- [ ] **Step 7:** Commit `chore: scaffold (M0)`.

**Acceptance:** clean-venv install works; `pytest` and `ruff` pass.

---

## M1 — Prove CPU fine-tuning

### Task 2: SST-2 smoke test + SetFit API verification

**Files:** Create `src/training/smoke_sst2.py`, `tests/test_smoke_api.py` (optional, marked `slow`); output `results/smoke_sst2.json`.

**Interfaces:** Produces verified facts about installed setfit API (record them in a comment block at top of `smoke_sst2.py` and in the JSON): `predict_proba` shape/type, label-string handling, `eval_strategy` arg name, `batch_size` arg on `predict`.

- [ ] **Step 1:** Write script: load `sst2` via `datasets`, `sample_dataset(train, label_column="label", num_samples=8, seed=42)`, 200-row eval slice, `SetFitModel.from_pretrained(base, labels=["negative","positive"])`, `TrainingArguments(batch_size=16, num_epochs=1, num_iterations=10, seed=42)`, `Trainer(...).train()` timed with `perf_counter`, `trainer.evaluate()`.
- [ ] **Step 2:** In the same script: assert `predict_proba(["a","b"]).shape == (2, 2)`; `save_pretrained(tmp)` → `from_pretrained(tmp)` → assert identical predictions on 20 texts and `allclose` probs; print `inspect.signature(TrainingArguments)` presence of `eval_strategy`.
- [ ] **Step 3:** Run `python -m src.training.smoke_sst2`. Expected: completes, writes `results/smoke_sst2.json` with `accuracy`, `train_seconds`, `hardware`.
- [ ] **Step 4:** Fix any API drift against the ROADMAP notes; if drift found, update the notes in ROADMAP.md rather than silently diverging.
- [ ] **Step 5:** Commit `feat: SetFit CPU smoke test (M1)`.

**Acceptance:** runs on CPU without errors, prints accuracy + train time, JSON saved.

---

## M2 — Dataset

### Task 3: Synthetic data generator + validator

**Files:** Create `src/data/generate.py`, `src/data/validate.py`, `tests/test_data_validate.py`; output `data/intents.csv`.

**Interfaces:**
- Produces: `generate(seed:int, per_intent:int=75) -> pd.DataFrame` (cols `text,intent`); `validate(df, intents) -> None` raising `ValueError` on: empty text, unknown label, exact duplicate, near-duplicate (normalized: lowercase, strip punctuation/whitespace), imbalance (any class != mean ±10%).
- `normalize(text:str)->str` shared helper in `src/data/validate.py` (reused by split + hard-test checks).

- [ ] **Step 1: Failing tests** for each validator failure mode (one tiny DataFrame per case) plus a happy path.
- [ ] **Step 2:** Run → FAIL. Implement `normalize` + `validate`. Run → PASS.
- [ ] **Step 3:** Write `generate.py`: per intent, 6–10 template families × entity pools (doctors, weekdays, dates, drugs, conditions) + hand-written colloquial/indirect lines; apply STT-style transform (lowercase/strip punctuation) to ~30%, prepend fillers ("um", "so", "can you") to ~20%; enforce length 3–30 words; dedupe by `normalize` and top up until 75/intent. Seeded RNG from config.
- [ ] **Step 2b:** Emergency intent gets extra variety (chest pain, can't breathe, overdose, stroke signs, bleeding, indirect phrasings) since recall matters most.
- [ ] **Step 3b:** Review sample: print 10 random rows per intent and read them; fix obviously wrong ones. Add `data/raw/README` noting provenance.
- [ ] **Step 4:** Run generate → `data/intents.csv`; `python -m src.data.validate` passes.
- [ ] **Step 5:** Commit `feat: synthetic dataset + validator`.

### Task 4: Split + hard test set

**Files:** Create `src/data/split.py`, `data/hard_test.csv` (hand-written, 40–60 rows, ≥5 per intent), `src/data/prepare.py` (orchestrates generate → validate → split), `tests/test_data_split.py`.

**Interfaces:** Produces `split(df, seed) -> (train, val, test)` stratified 70/15/15; `prepare.main()` writes `data/processed/{train,val,test,hard_test}.csv`. Consumes `validate`, `normalize`.

- [ ] **Step 1: Failing tests:** (a) zero overlap by exact text and by `normalize(text)` among train/val/test; (b) `hard_test` has zero overlap with all three; (c) per-class proportions within ±1 row of 70/15/15; (d) hard_test labels all valid.
- [ ] **Step 2:** Run → FAIL; implement `split.py` (`train_test_split` stratified twice, seeded) and `prepare.py`; hand-write `hard_test.csv` (multi-intent / ambiguous, e.g. the roadmap example → label best single intent; include a few low-confidence emergencies).
- [ ] **Step 3:** `make data` → files exist; tests PASS.
- [ ] **Step 4:** Add README "Dataset" stub (synthetic caveat). Commit `feat: splits + hard test (M2)`.

**Acceptance:** `make data` works, validate passes, leakage test passes, README dataset section drafted.

---

## M3 — Baseline and fine-tuning

### Task 5: Baseline + fine-tune + `IntentRouter`

**Files:** Create `src/training/baseline.py`, `src/training/train_setfit.py`, `src/models/router.py`, `src/evaluation/metrics.py` (shared: `compute_metrics(y_true,y_pred,intents)->dict` incl. accuracy, macro P/R/F1, per-intent F1, `emergency_recall`, confusion matrix), `tests/test_router.py`, `tests/test_metrics.py`.

**Interfaces:**
- `IntentRouter.load(path:str) -> IntentRouter`
- `IntentRouter.predict(text:str) -> {"intent":str,"confidence":float,"probs":dict[str,float]}`
- `IntentRouter.predict_batch(texts:list[str]) -> list[dict]` (same schema)
- `IntentRouter` raises `ValueError` on empty/whitespace text.
- Probability dict keys ordered per `intents.yaml`, summing to ~1.

- [ ] **Step 1: Failing tests** (`tests/test_metrics.py`): hand-built y_true/y_pred → known accuracy, macro F1, `emergency_recall`. Run → FAIL → implement `metrics.py` → PASS.
- [ ] **Step 2:** `baseline.py`: frozen `SentenceTransformer` embeddings of train → `LogisticRegression(max_iter=1000)`; evaluate on test via `compute_metrics`; write `results/baseline.json`.
- [ ] **Step 3:** `train_setfit.py`: read `configs/training.yaml`, `SetFitModel.from_pretrained(base, labels=INTENTS)`, `TrainingArguments(**cfg.setfit, seed=seed)`, `Trainer(train, val)`, time training, `save_pretrained(paths.model)` explicitly, write `results/train_log.json` (train_seconds, hyperparams, val metrics). Tune hyperparameters on **val only** (small grid over `num_iterations` ∈ {5,10,20}, `num_epochs` ∈ {1,2}); record grid in the log; fix the chosen values in config. Test evaluation happens once, in `evaluate.py` (M4) or at the end of this script into `results/finetuned.json`.
- [ ] **Step 4: Failing router tests** (use a session-scoped fixture that trains a tiny model on 5/intent from `train.csv`, 1 iter, so tests stay fast): schema keys, `0 ≤ confidence ≤ 1`, probs sum ≈ 1, `predict_batch` equals per-item `predict`, empty text → `ValueError`, save→load reproduces predictions.
- [ ] **Step 5:** Run → FAIL; implement `router.py`; PASS.
- [ ] **Step 6:** `make train` → `models/setfit_router` + `results/finetuned.json`; compare macro F1 vs baseline honestly (note result either way).
- [ ] **Step 7:** Commit `feat: baseline, SetFit training, IntentRouter`.

### Task 6: Data-size ablation

**Files:** Create `src/training/ablation.py`; outputs `results/ablation.json`, `results/ablation.png`.

**Interfaces:** Consumes `sample_dataset`, `compute_metrics`, training config. Produces `{size: {macro_f1_mean, macro_f1_std, accuracy_mean, accuracy_std, emergency_recall_mean, per_seed:[...]}}`.

- [ ] **Step 1:** Loop sizes [10,25,50,75] × seeds [42,43,44]; each run: `sample_dataset(train, "label", num_samples=n, seed=s)`, fresh model, train, evaluate on test. (Train split has ~52/intent, so cap n at available count per class and **log the effective n** — note 75 > ~52 available in train; either generate more data, e.g. `per_intent: 110`, or define the sizes against train and say so. **Decision needed before running:** raise `per_intent` to ~110 so train ≥ 75/intent, or document the cap.)
- [ ] **Step 2:** Save JSON + errorbar plot (matplotlib, labeled axes). Run `python -m src.training.ablation` (12 trainings; expect tens of minutes on CPU).
- [ ] **Step 3:** Commit `feat: ablation (M3 done)`.

**Acceptance:** `baseline.json`, `finetuned.json`, ablation 4×3 table saved.

---

## M4 — Evaluation harness

### Task 7: `evaluate.py` + `benchmark.py`

**Files:** Create `src/evaluation/evaluate.py`, `src/evaluation/benchmark.py`, `src/evaluation/banner.py` (`print_banner(fields: dict)`), `tests/test_evaluate.py`.

**Interfaces:** CLI `--model <path|baseline> --split test|hard_test`; baseline must be loadable through the same `IntentRouter`-like interface (add `src/models/baseline_router.py` or a `BaselineRouter` with same `predict_batch` signature; save the LR head via `joblib` in `models/baseline/`). Outputs `results/eval_<name>.json`, `results/confusion_<name>.png`, `results/bench_<name>.json`.

- [ ] **Step 1: Failing test:** banner formatting for a sample dict matches the roadmap format exactly (golden string).
- [ ] **Step 2:** Implement `banner.py` → PASS.
- [ ] **Step 3:** `evaluate.py`: load model, predict test, `compute_metrics`, save confusion PNG, print banner + separate `EMERGENCY recall:` line.
- [ ] **Step 4:** `benchmark.py`: `torch.set_num_threads(cfg.threads)` (add `benchmark.threads` to config; log it), 50 warm-ups, 1000 single-request timings (`perf_counter`, cycle through test texts), report P50/P95/P99/mean/throughput, model size on disk (sum of file sizes), RSS via `psutil` after load, train time from `results/train_log.json`, hardware info.
- [ ] **Step 5:** Run both for `models/setfit_router` and the baseline; commit `feat: evaluation + benchmark (M4)`.

**Acceptance:** both commands run, write JSON, banner prints; baseline vs fine-tuned comparable.

---

## M5 — Confidence routing + LLM fallback

### Task 8: LLM clients + calibration

**Files:** Create `src/llm/client.py`, `src/llm/anthropic_client.py`, `src/llm/mock_client.py`, `src/models/calibration.py`, `tests/test_llm_clients.py`, `tests/test_calibration.py`; outputs `results/threshold_sweep.json`, `results/threshold_sweep.png`, updated `configs/routing.yaml`.

**Interfaces:**
- `class LLMClient(Protocol): classify_intent(text:str, intents:list[str]) -> dict` → `{"intent":str,"reason":str,"input_tokens":int,"output_tokens":int,"failed":bool}`; and `generate(prompt:str) -> str` (agents' phrasing).
- `parse_llm_intent(raw:str, intents) -> dict` in `client.py`: strips code fences, `json.loads`, validates intent ∈ intents, else returns `GENERAL_QUERY` with `failed=True`.
- `MockLLMClient(delay_s:float=0, rules:dict|None=None)` — deterministic keyword rules, token counts fake-but-fixed (marked `mock` so they are never reported as real cost).
- `sweep_thresholds(probs_conf, correct, thresholds)->list[dict]` (coverage, covered_accuracy); `choose_threshold(sweep, target)->float` = smallest threshold with covered_accuracy ≥ target (fallback: highest-accuracy point, documented).

- [ ] **Step 1: Failing tests:** `parse_llm_intent` on valid JSON, fenced JSON, prose, invalid label, empty string; `MockLLMClient` determinism and that `classify_intent` never raises; `choose_threshold` on a hand-built sweep (including the no-threshold-meets-target case); `AnthropicLLMClient` with an injected fake SDK client that raises → returns `failed=True` fallback.
- [ ] **Step 2:** Run → FAIL; implement; PASS. Anthropic client reads `ANTHROPIC_API_KEY`, model from routing config, strict-JSON prompt, `timeout` from config.
- [ ] **Step 3:** `calibration.py` CLI: predict on **val**, sweep 0.30–0.95 step 0.05, save JSON/PNG, write chosen `threshold` into `configs/routing.yaml` along with `threshold_rule` text.
- [ ] **Step 4:** Run calibration; commit `feat: LLM clients + threshold calibration`.

### Task 9: Hybrid routing logic

**Files:** Create `src/systems/base.py`, `src/systems/hybrid.py`, `src/systems/minilm_only.py`, `tests/test_routing.py`.

**Interfaces:**
- `RoutingRecord = TypedDict(intent, confidence, source: "minilm"|"llm"|"emergency_rule", llm_called: bool, latency_ms: float, llm_failed: bool, input_tokens: int, output_tokens: int)`
- `class RoutingSystem(Protocol): route(text:str) -> RoutingRecord`
- `HybridSystem(router: IntentRouter, llm: LLMClient, cfg: dict)`; `MiniLMOnlySystem(router, cfg)` (emergency rule still applied).
- Shared helper `apply_emergency_rule(probs, emergency_floor) -> bool`.

- [ ] **Step 1: Failing tests** (fake router returning scripted probs; `MockLLMClient` with call counter): (a) conf ≥ threshold → `llm_called False`, source minilm; (b) conf < threshold → LLM called, intent from LLM; (c) EMERGENCY top → emergency, no LLM; (d) EMERGENCY prob 0.35 but top is GENERAL_QUERY at 0.5 → emergency; (e) EMERGENCY prob 0.29 → not triggered; (f) LLM returns garbage → `GENERAL_QUERY`, `llm_failed True`; (g) LLM raises → same; (h) threshold read from config, not hardcoded.
- [ ] **Step 2:** Run → FAIL; implement; PASS.
- [ ] **Step 3:** Commit `feat: hybrid routing (M5)`.

**Acceptance:** unit tests pass; threshold justified by sweep JSON.

---

## M6 — Agents and LangGraph

### Task 10: Mock tools + agents

**Files:** Create `src/agents/tools.py`, `booking.py`, `prescription.py`, `research.py`, `emergency.py`, `faq.py`, `tests/test_tools.py`, `tests/test_agents.py`.

**Interfaces:**
- `ToolBox(seed:int=42)`: in-memory fake doctors/slots/patients/prescriptions; methods `check_slots(doctor=None, day=None)`, `book_appointment(patient_id, doctor, slot)`, `reschedule_appointment(patient_id, new_slot)`, `cancel_appointment(patient_id)`, `get_prescription(patient_id)`, `get_followup(patient_id)`, `search_pubmed(query)`, `search_web(query)` (canned; real PubMed behind config flag `research.use_pubmed: false`). All return plain dicts; errors as `{"error": "..."}` — no exceptions.
- Each agent: `class Agent: name:str; run(text:str, toolbox, llm:LLMClient|None) -> {"agent": str, "tool_calls": list[{"name","args"}], "response": str}`. Names: `booking_agent, prescription_agent, research_agent, emergency_agent, faq_agent`.
- `emergency_agent` makes zero tool calls, fixed safety message.
- `BookingAgent` takes the routed `intent` (book/reschedule/cancel) as a parameter: `run(text, toolbox, llm, intent)`; simple regex extraction of doctor/day.

- [ ] **Step 1: Failing tests:** each tool happy path + error path (reschedule with no booking, booking a taken slot, unknown patient); each agent invoked with `llm=None` and with `MockLLMClient`; emergency agent has `tool_calls == []` and message mentions emergency services.
- [ ] **Step 2:** Run → FAIL; implement; PASS.
- [ ] **Step 3:** Commit `feat: mock tools and agents`.

### Task 11: LangGraph workflow + demo

**Files:** Create `src/graph/state.py`, `src/graph/workflow.py`, `demo.py`, `tests/test_graph.py`.

**Interfaces:**
- `class GraphState(TypedDict): text:str; routing:dict; agent_name:str; tool_calls:list; response:str`
- `build_graph(system: RoutingSystem, toolbox, llm) -> CompiledGraph`; `run_text(graph, text) -> GraphState`.
- Intent→agent map in one place: `INTENT_TO_AGENT` in `workflow.py`.

- [ ] **Step 1: Failing test:** parametrized over all 8 intents with a stub `RoutingSystem` returning a fixed intent → asserts expected `agent_name` and, where applicable, expected first tool-call name (e.g. RESCHEDULE → `reschedule_appointment`, EMERGENCY → none).
- [ ] **Step 2:** Run → FAIL; implement `StateGraph` (START → `intent_router` → conditional edges → agent nodes → END); PASS.
- [ ] **Step 3:** `demo.py "…"` prints intent, confidence, agent, tool calls, response; uses real model if `models/setfit_router` exists, mock LLM unless API key set.
- [ ] **Step 4:** Run `python demo.py "Can I move my appointment to Friday?"`; commit `feat: LangGraph workflow (M6)`.

---

## M7 — Compare three systems

### Task 12: LLM-only system + e2e tasks + comparison

**Files:** Create `src/systems/llm_only.py`, `src/evaluation/e2e_tasks.py`, `src/evaluation/compare_systems.py`, `tests/test_compare.py`; outputs `results/compare_systems.json`, `results/compare_systems.md`.

**Interfaces:**
- `LLMOnlySystem(llm, cfg).route(text) -> RoutingRecord` (emergency rule N/A; LLM decides — note this in README).
- `E2E_TASKS: list[{"text","expected_intent","expected_agent","expected_tool"|None}]`, ~40 entries, ≥8 from `hard_test.csv`.
- `task_success(state, task) -> bool` = correct agent **and** correct tool call.
- `cost_per_request(records, pricing) -> float` from actual token usage (MiniLM-only = 0).

- [ ] **Step 1: Failing tests:** `task_success` cases; `cost_per_request` arithmetic; with no API key `compare_systems` marks LLM-only fields `"not_run"` and never calls the mock for A; hybrid with mock fallback labeled `"llm_backend": "mock"` in JSON so mock-derived hybrid numbers are clearly flagged.
- [ ] **Step 2:** Run → FAIL; implement; PASS.
- [ ] **Step 3:** `compare_systems.py`: for each of {test, hard_test, e2e} run A/B/C; metrics: intent accuracy, macro F1, P50/P95 latency, CPU usage (`psutil.Process().cpu_percent` over run / `process_time` per request), cost/request, LLM calls, e2e task success. Render markdown table.
- [ ] **Step 4:** Run (note whether API key present); commit `feat: three-system comparison (M7)`.

**Acceptance:** JSON + md with all fields filled or `not_run`.

---

## M8 — API and voice

### Task 13: FastAPI `/chat`

**Files:** Create `src/api/app.py`, `tests/test_api.py`.

**Interfaces:** `POST /chat {"text": str (min_length=1, strip, max_length from config)}` → `{intent, confidence, source, agent, response, latency_ms}`; `GET /health` → `{"status":"ok","model_loaded":bool}`. Model, hybrid system, graph built once in a `lifespan` handler; test overrides via dependency/`app.state` with fake router + mock LLM.

- [ ] **Step 1: Failing tests (`TestClient`):** `/health` 200; `/chat` happy path schema; empty/whitespace text → 422; over-length text → 422; emergency utterance routes to `emergency_agent`.
- [ ] **Step 2:** Run → FAIL; implement; PASS.
- [ ] **Step 3:** Manual check: `uvicorn src.api.app:app` + `curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"text":"Can I move my appointment to Friday?"}'`.
- [ ] **Step 4:** Commit `feat: FastAPI endpoint (M8)`.

### Task 14: Voice adapter interface (optional, last)

**Files:** Create `src/voice/adapters.py` (`STTAdapter`, `TTSAdapter` protocols + `TextSTT`/`TextTTS` no-op implementations), `requirements-voice.txt` (LiveKit, stretch), `tests/test_voice.py`.

- [ ] **Step 1:** Test: text adapters round-trip a string through `voice_turn(stt, graph, tts, audio_or_text)`; implement; commit. Skip LiveKit unless time allows.

---

## M9 — Polish

### Task 15: Results renderer, README, final verification

**Files:** Create `scripts/render_results.py`, `README.md`; modify `Makefile`.

- [ ] **Step 1:** `render_results.py` reads `results/*.json` and emits markdown tables (baseline vs fine-tuned, ablation, latency, threshold, comparison); missing files render as `TBD`. Test with a fixture JSON.
- [ ] **Step 2:** README: overview, Mermaid architecture diagram, setup, make commands, dataset + synthetic caveat, generated results, design decisions (SetFit, threshold, emergency floor), limitations (synthetic/small data, in-distribution test inflates accuracy, mock tools, single-machine latency + hardware info), future work, resume summary with `XX` placeholders.
- [ ] **Step 3:** Fresh-clone check in a temp dir/venv: `make data && make train && make evaluate && make benchmark && make compare`; `ruff check .` and `pytest` clean.
- [ ] **Step 4:** Tick the Definition of Done checklist in ROADMAP §8; commit `docs: README + results (M9)`.

---

## Self-Review notes

- **Coverage:** all hard rules map to tasks (rule 8 → Tasks 5/9/13; rule 2 → Tasks 5/8 use val only; rule 5 → Task 7).
- **Flagged issue found in spec:** 70/15/15 of ~600 rows gives ~52 train examples/intent, but the ablation requires 75/intent. Resolved by deciding in Task 6 (raise `per_intent` to ~110, then the 75 point is valid). Ask the user if they'd rather keep 600 rows.
- **Cost/time:** ablation (12 trainings) and hyperparameter grid are the slow CPU steps; keep `num_iterations` small.
- **LLM-only needs `ANTHROPIC_API_KEY`**; otherwise `not_run` per spec.
