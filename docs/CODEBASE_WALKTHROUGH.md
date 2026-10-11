# Codebase Walkthrough (big picture)

This is a guided tour of the repo for orientation purposes. It's deliberately high-level; use it as a map, then ask
follow-up questions on whichever part you want to go deeper into. (The repo also has `docs/ARCHITECTURE.md` and
`docs/WALKTHROUGH.md` with a real traced run — this doc is the "read me first" layer above both.)

---

## 1. What this project actually is

**"Research Intelligence OS"** — an agentic research assistant (think: "ask it a question, it searches the web,
reads sources, and gives you a cited report") built around one central idea:

> Every claim in the final report must carry a **verbatim quote** that provably exists in a stored source. If a
> claim isn't backed by an exact quote, it's dropped before it ever reaches the user.

The second design idea is cost control: most requests should **never touch a large LLM**. A tiny, fine-tuned
classifier (not a prompt-based LLM call) decides what kind of request this is, in ~11ms on a CPU, and only escalates
to an LLM when it's unsure.

So this is really three things living in one repo:
1. A **router** (small ML classifier) that cheaply classifies research intents.
2. An **agent pipeline** (LangGraph-based) that plans and executes multi-step research using LLM calls + tools.
3. A **harness** — the operational scaffolding around the agents (budgets, retries, approvals, record/replay,
   verification gates, tracing, memory) that makes the agent pipeline safe and auditable rather than a loose script.

There's also a **FastAPI backend + Next.js frontend** so you can run this as a real app, and a **voice** layer
(LiveKit) bolted on as a secondary interface. A `legacy/` folder holds an earlier healthcare-domain version of the
same idea, kept for reference/tests.

---

## 2. The end-to-end flow (one request)

```
request
  -> ROUTER        fine-tuned MiniLM classifier -> intent + confidence + entities (no LLM, ~11ms)
  -> SUPERVISOR     confident? go straight to a workflow template (DIRECT, 0 LLM calls)
                    unsure? one schema-validated LLM call to pick a workflow (LLM_FALLBACK)
  -> PLANNER        expands the chosen workflow template into concrete steps (search/read/extract/verify/synthesize),
                    validates it (no cycles, known tools, within budget), estimates cost
                    [if plan is "expensive" -> pause for human approval (LangGraph interrupt)]
  -> EXECUTOR       runs steps in dependency order, through the harness (every tool call checked against
                    registry/budgets), never raises — failures are captured as data
       -> search_agent     builds queries, calls web_search / github_search / pubmed / arxiv, ranks & dedups
       -> source_agent     fetches full pages (fetch_url), picks best passages, strips prompt-injection text
       -> evidence_agent   one LLM call turns passages into atomic claims, each with a verbatim quote
       -> verifier         deterministic checks first (quote exists verbatim, source exists, dates, numbers
                            match) + one LLM judge call for entailment + contradiction detection -> PASS/FAIL gate
       -> synthesis_agent  writes findings using ONLY verified ("SUPPORTED") claims, cites claim ids
  -> RECOVERY       on any failure/verifier-FAIL: retry, try alternate tool, broaden query, replan once, ask the
                    user, or terminate safely into a PARTIAL artifact — not a crash
  -> REPORT         ResearchReport (JSON / Markdown / PDF), citations enforced, served via API and shown in the UI
```

This whole thing is one compiled **LangGraph state graph** (`research_os/graph/workflow.py`), so each box above is a
graph node, and the arrows with recovery loops are conditional edges.

---

## 3. Repository layout (what lives where)

```
research_os/        <- the actual system (Python package)
  router/              the small ML classifier (training, inference, calibration, evaluation)
  llm/                 LLM client abstraction (Gemini + a deterministic offline "demo" LLM for CI/UI)
  agents/              supervisor, planner, and 5 step-agents (search/source/evidence/verifier/synthesis)
  harness/             budgets, tool registry enforcement, recovery policy, approvals, record/replay, tracing hooks
  graph/               LangGraph wiring: state shape, workflow graph, executor, step templates
  tools/               the actual tools agents can call (web_search, fetch_url, github, pubmed, arxiv) + registry
  memory/              persistent storage (SQLAlchemy models, Alembic migrations) for past runs/findings, embeddings
  artifacts/           turns verified claims into a ResearchReport; renders Markdown (Jinja2) and PDF (WeasyPrint)
  observability/       Langfuse tracing hook (no-op if no keys configured)
  api/                 FastAPI app: POST /runs, SSE event stream, reports, approve/reject endpoints
  voice/               LiveKit voice agent + text/audio adapters (secondary interface, not the primary demo path)
  prompts/             every LLM prompt as a versioned .md file (planner.v1.md, evidence.v2.md, etc.)
  contracts/           the Pydantic data contracts (RouterOutput, ResearchStep, ExecutionBudget, etc.) shared by
                       everything above — this is the glue that keeps modules honest about data shape
  legacy/              the original (M6-era) healthcare-domain demo; kept for history/tests, not the main path

configs/             YAML config for nearly everything: intents, routing threshold, workflow templates, budgets,
                     tool permissions, recovery rules, memory reuse policy, model-to-role mapping, tracing, API
evaluation/          the offline evaluation harness: 120 tasks, frozen "cassette" recordings of web responses,
                     4 systems (A/B/C/D) compared, LLM judges, failure-injection suite, report generation
experiments/         output of the evaluation/training runs (the numbers quoted in README come from here)
models/               trained router artifacts (SetFit MiniLM weights) — git-ignored normally, present here
migrations/           Alembic DB migration scripts for the memory store
scripts/              small CLI utilities (dump OpenAPI schema, render results tables, demo GIF, tool smoke tests)
frontend/            Next.js + Tailwind UI: workspace, live execution graph (React Flow), evidence explorer with
                     quote highlighting, report viewer, run history, approval modal
tests/               ~600 tests: unit/ (per-module) and integration/ (full pipelines)
docker/              docker-compose + Dockerfile for one-command `docker compose up`; optional self-hosted Langfuse
runs/                 on-disk record of past runs (events.jsonl, recording.jsonl for replay, run.json) — local state
docs/                architecture notes, this walkthrough, voice notes, migration notes, archived roadmaps
```

---

## 4. The router — a small classifier, not an LLM

Location: `research_os/router/`

* It's a **SetFit-fine-tuned MiniLM** sentence embedding model + a small classification head (`model.safetensors` /
  `model_head.pkl` in `models/router/`), not a prompt to a big LLM.
* Classifies a request into one of **8 research intents** (e.g. `FACT_LOOKUP`, `COMPARISON`, `DEEP_DIVE`, etc. — see
  `configs/intents.yaml`).
* Also runs **deterministic entity extraction** (gazetteer + regex, `router/entities.py`) — no LLM, no inference
  randomness.
* Outputs a `RouterOutput` (intent, confidence, entities, routing_mode, latency_ms).
* **Routing mode decision**: if confidence is above a threshold (0.60, chosen on a validation sweep — see
  `router/calibration.py`), mode = `DIRECT` (supervisor skips the LLM entirely). Below it, mode = `LLM_FALLBACK`
  (supervisor makes exactly one schema-validated LLM call to pick a workflow).
* There's a whole **training pipeline** (`router/training/`) and **evaluation pipeline**
  (`router/evaluation/`) that produces the comparison tables in the README (fine-tuned vs frozen-embedding baseline,
  latency benchmarks, data-size ablation). `make data && make train` regenerates it from synthetic + LLM-generated
  data (`router/data/`).

Why it matters: this is the thing that makes "most requests are cheap" true. The supervisor only pays for an LLM
call when the router is unsure.

---

## 5. Supervisor, planner, and the 5 step-agents

Location: `research_os/agents/`

* **`supervisor.py`** — takes the router's output and decides which *workflow template* to run (e.g.
  `fact_lookup`, `comparison`, `research_monitoring`...). DIRECT mode maps intent -> template deterministically.
  LLM_FALLBACK mode asks the LLM to pick one, validates the answer against a schema, and falls back to
  `general_research` if the LLM's answer doesn't validate. Also checks memory for a "recall" shortcut (if a very
  similar question was answered recently, it can answer from memory with no new research).
* **`planner.py`** — expands the chosen workflow template (`configs/workflows.yaml`) into a concrete
  `ResearchStep` list for the specific entities found, validates the plan (no cycles, tools exist, within budget),
  computes a cost estimate (searches/sources/LLM calls/tool calls), and can `replan()` once if execution fails.
* The **five step-agents**, each doing one job and each described by a versioned prompt file in
  `research_os/prompts/`:
  | Agent | File | Tools it may use | Job |
  |---|---|---|---|
  | search_agent | `search_agent.py` | web_search, github_search, pubmed, arxiv | build queries, rank/dedup sources |
  | source_agent | `source_agent.py` | fetch_url | fetch full pages, pick best passages, strip prompt-injection text |
  | evidence_agent | `evidence_agent.py` | none (1 LLM call) | turn passages into atomic claims + verbatim quotes |
  | verifier | `verifier.py` | none | deterministic checks + 1 LLM judge call + conflict detection -> PASS/FAIL |
  | synthesis_agent | `synthesis_agent.py` | none | write findings from SUPPORTED claims only, with citations |

* Every LLM call anywhere in this stack is validated against a **Pydantic schema** with at most 2 retries
  (`llm/structured.py`). Models are only ever shown short local ids (`c1`, `s2`, ...) for claims/sources — never raw
  hashes — so they can't mis-copy an id.

---

## 6. The verifier — the "claims must be grounded" gate

Location: `research_os/agents/verifier.py` + `research_os/harness/verification.py`

This is the component the whole project is built to showcase, so it's worth calling out separately:

1. **Deterministic checks first** (cheap, auditable, can't be argued with by an LLM):
   - the cited source actually exists
   - the quote exists **verbatim** in the stored source text
   - source type matches what's claimed, dates are consistent
   - comparative/numeric/universal/causal language in a claim must actually appear in its quote (e.g. a claim can't
     say "always" if the quote doesn't support that strength)
2. **LLM judge** only runs afterward, for entailment (does the quote actually support the claim's meaning) — and it
   **cannot upgrade** a claim that the deterministic checks already rejected.
3. **Contradiction detection across sources** — conflicts are surfaced in the report, not silently hidden, and not
   automatically fatal to the run (a conflict confirmation prompt, `prompts/conflict_confirm.v2.md`, requires two
   genuinely different sources talking about the same subject before it's flagged as a real conflict — this was a
   bug fix after an earlier version flagged a false conflict and dropped a correct answer).
4. **Claims that fail -> never reach synthesis.** The synthesis agent only ever sees `SUPPORTED` claims.

---

## 7. The harness — what makes this "production-shaped" rather than a prompt-chaining script

Location: `research_os/harness/` (+ `tools/` for the registry)

| Piece | File | What it does |
|---|---|---|
| Tool registry | `tools/registry.py` | every tool call is schema-checked, permission-checked, domain-policy-checked, SSRF-protected, timed out, retried, output-capped |
| Budgets | `harness/budgets.py` | hard limits (steps, tool calls, searches, LLM calls, wall-clock seconds) checked **before** every step/tool/LLM call, not just at the end |
| Recovery | `harness/recovery.py` | a lookup table (`configs/recovery.yaml`) mapping failure types -> actions: retry, retry-with-error, broaden query, alternate tool, skip source, replan once, ask user, terminate safely |
| Approvals | `harness/approvals.py` | LangGraph `interrupt()` — pauses execution for a human decision when a plan looks expensive, resumable |
| Record/replay | `harness/replay.py` + recording in `runs/<id>/recording.jsonl` | every tool call and LLM call is recorded; a run can be replayed byte-for-byte with no network access — this is what makes the evaluation suite deterministic |
| Fault injection | `harness/faults.py` | used by the offline failure-injection evaluation to simulate timeouts, garbage tool output, malformed LLM output, etc. |
| Verification | `harness/verification.py` | the deterministic half of the verifier (see §6) |

The executor (`graph/executor.py`) runs steps in dependency order and **never raises** — any failure becomes a typed
`FailureType` that the recovery table reacts to, so a bad tool response degrades into a retry or a `PARTIAL` report
instead of crashing the whole run.

---

## 8. Memory, artifacts, observability

* **Memory** (`research_os/memory/`) — SQLAlchemy models + Alembic migrations (`migrations/`), SQLite by default,
  Postgres verified. Stores past runs/findings so the supervisor can short-circuit ("recall") near-duplicate
  questions, and so a "monitoring" workflow can diff new findings against prior ones. Embeddings
  (`memory/embeddings.py`) are brute-force cosine over MiniLM vectors — no vector DB, a deliberate simplicity choice
  (see README "Design decisions").
* **Artifacts** (`research_os/artifacts/`) — builds the canonical `ResearchReport` object (citation integrity is
  enforced structurally, not just by convention), then renders it to Markdown (Jinja2 templates) or PDF
  (WeasyPrint).
* **Observability** (`research_os/observability/`) — a Langfuse event listener wired into the event bus; it's a
  no-op if no Langfuse keys are set, so it never blocks local/offline use.

---

## 9. API and frontend

* **API** (`research_os/api/`, FastAPI):
  - `app.py` wires everything together at startup (picks Gemini vs the offline demo LLM, builds the router,
    agents, tool registry, memory store).
  - `routes_runs.py` / `service.py` — `POST /runs` to start a run, `GET /runs` / `GET /runs/{id}` for status, an
    SSE stream (resumable) for live events, report endpoints (`/runs/{id}/report?format=json|markdown|pdf`), and
    approve/reject endpoints for the human-in-the-loop interrupts.
  - `schemas.py` — Pydantic response models (also the source of the OpenAPI schema consumed by the frontend).
* **Frontend** (`frontend/`, Next.js + Tailwind + React Flow):
  - `src/app/page.tsx` — the main workspace (submit a request).
  - `src/app/runs/[id]/page.tsx` — a single run: live execution graph, budget meters, event log.
  - `src/components/RunGraph.tsx` — renders the LangGraph execution as a live graph.
  - `src/components/EvidenceExplorer.tsx` / `SourceViewer.tsx` — shows each claim next to its quote, highlighted
    inside the actual stored source text (the UI's proof that nothing is made up).
  - `src/components/ReportView.tsx` — the final report with clickable citations.
  - `src/components/ApprovalDialog.tsx` — the UI side of the human-approval interrupt.
  - `src/api/schema.d.ts` is **generated** from the backend's OpenAPI schema (`make gen-api-types`), so frontend
    types stay in sync with the API automatically.

---

## 10. Voice layer

Location: `research_os/voice/`

A secondary interface on top of the same pipeline: `adapters.py` / `legacy_adapters.py` turn text results into
spoken summaries, `gemini_audio.py` / `audio.py` handle audio I/O, and `livekit_agent.py` runs it as a LiveKit voice
agent. Per the README, this was demonstrated end-to-end with a bot caller speaking synthesized audio through a real
LiveKit server, but real microphones/accents/noise are explicitly untested (`docs/VOICE.md` has details). This is
clearly not the primary demo path — the main story is the text pipeline + UI.

---

## 11. Evaluation — how the README's numbers are produced

Location: `evaluation/` (+ output in `experiments/`)

This is a serious, separate harness for measuring the system, not just unit tests:

* **120 fixed tasks**, each with intent labels and gold claim labels (written by the project author).
* **Frozen "cassette" recordings** of real web responses (`evaluation/cassettes/`) so replays are deterministic and
  free — recorded once live (`make eval-record`, costs API calls), then replayed forever (`make eval`).
* **Four systems compared**: A = LLM-only router, B = MiniLM-only router, C = hybrid, D = hybrid + full harness
  (verifier, recovery, budgets). There are also harness **ablations** (no-verifier, no-recovery, no-budgets) to
  isolate each component's effect.
* **LLM judges** (`judges.py`) score quality dimensions (unsupported-claim rate, citation correctness, source
  relevance) using a *different* model from the one used inside the pipeline, with measured agreement against hand
  labels reported honestly in the README (not claimed as ground truth).
* **Failure injection** (`failure_injection.py`) — an offline suite of 11 synthetic failure scenarios (timeouts,
  garbage tool output, malformed LLM output, budget exhaustion, prompt injection, etc.) checking that the harness's
  recovery table actually does the right thing.
* `report_results.py` / `scripts/render_results.py` turn all of this into the tables embedded in README.md between
  `<!-- RESULTS:START/END -->` markers — "nothing is typed by hand."

The README's "Limitations" section is unusually candid about this evaluation's weaknesses (single annotator, modest
judge agreement kappa=0.33, frozen replay != live web, weak real-task recovery) — worth reading once you get there.

---

## 12. Tests

`tests/unit/` (one file per module, ~ matches the `research_os/` tree) and `tests/integration/` (full pipeline runs,
e.g. `test_comparison_workflow.py`, `test_memory_pipeline.py`). `make test` runs everything offline, no network, no
API keys needed (thanks to the demo LLM + fixture tools). ~600 tests per the README badge.

---

## 13. How to actually run it (for reference)

```bash
docker compose up --build        # one command, needs .env with GOOGLE_API_KEY + TAVILY_API_KEY
# or, locally:
make data && make train           # trains the router (~1hr CPU) — or just use the checked-in models/ dir
make api                          # FastAPI on :8000 (falls back to offline demo LLM without a key)
make ui                           # Next.js on :3000
```

---

## 14. Suggested order for follow-up deep-dives

Given the architecture, these are natural next topics (pick any):
1. **Router internals** — SetFit training, calibration/threshold selection, entity extraction.
2. **Planner + workflow templates** — how `configs/workflows.yaml` turns into a `ResearchStep` plan.
3. **Verifier deep-dive** — the deterministic checks + the LLM judge + conflict confirmation logic.
4. **Harness: budgets/recovery/approvals** — the exact failure-type -> action table.
5. **Record/replay + the evaluation harness** — how determinism is achieved for 120 tasks.
6. **API + SSE event model** — how the frontend gets live updates and resumes a stream.
7. **Memory store** — schema, recall/reuse policy, monitoring diff.
8. **Frontend** — React Flow execution graph, evidence explorer, approval modal.
9. **Voice layer** — LiveKit agent wiring.
10. **The `legacy/` healthcare demo** — what it was and why it's still around.
