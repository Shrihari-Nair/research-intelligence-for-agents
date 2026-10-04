# ROADMAP PART 2: Research Intelligence OS

> Instructions for Claude Code: this file continues `ROADMAP.md`. The repo already contains the MiniLM + SetFit router, the hybrid confidence routing with LLM fallback, and the LangGraph workflow with simple agents (original M0 to M6). Do **not** restart anything. Read this whole file, inspect the existing repo, then work milestone by milestone in order. Do not start a milestone until the previous one's acceptance criteria pass. Commit after each milestone.
>
> The original roadmap's M7 (compare systems), M8 (API) and M9 (polish) are superseded by this file. Their ideas are folded in below (M6.1 baseline, M15.5 API, M20 evaluation).

(Saved verbatim from the user's message; condensed only in this header. Sections 0-9 below are the user's text.)

---

## 0. Target system

A research system where a fine-tuned MiniLM router is a low-latency control point, an LLM supervisor and planner turn a request into a multi-step plan, specialized agents gather and analyze evidence, a verifier gates what reaches the report, and a harness governs everything (tools, budgets, retries, recovery, approvals, replay, observability).

```
USER
  │
  ▼
MiniLM Router (fine-tuned) ──► intent, confidence, entities, routing_mode
  │
  ▼
Supervisor
  ├─ DIRECT mode (high confidence): picks the workflow template for the intent, no LLM call
  └─ LLM mode (low confidence): LLM decides workflow / intent
  │
  ▼
Planner ──► ResearchPlan (steps with dependencies)
  │
  ▼
Executor (LangGraph) runs steps through the harness
  Search Agent ─► Source Analysis Agent ─► Evidence Agent
  │
  ▼
Verifier (SUPPORTED / CONTRADICTED / UNSUPPORTED)
  ├─ PASS ─► Synthesis ─► Research Artifact ─► API / Frontend
  └─ FAIL ─► Recovery (retry, alternate tool, replan, ask user, terminate safely)

HARNESS around everything: policies, budgets, tool permissions, schema validation,
retries, timeouts, evidence requirements, verification, human approval, replay,
evaluation, observability
```

Design decision: the Supervisor always runs. In DIRECT mode it is deterministic (intent maps to a workflow template). In LLM mode it calls the LLM. One code path; "LLM calls saved by the router" is measurable.

## 1. Hard rules (carried over and extended)

1. **Never fabricate numbers.** Every metric in docs comes from a script run saved under `experiments/` or `results/`.
2. **Never touch the test sets** during training, threshold tuning, prompt tuning, or hyperparameter selection. Use validation.
3. **Config-driven everything** (`configs/*.yaml`): intents, thresholds, budgets, model names per role, tool settings, paths. No model names or magic constants in code.
4. **Fixed seeds** and logged hardware info for every experiment.
5. **All LLM output is validated** against a Pydantic schema. Invalid output triggers a bounded retry (max 2), then the failure is classified and handed to recovery. No regex-parsing of free text.
6. **Every run is bounded** by an `ExecutionBudget`. No unbounded loops anywhere. Check the budget before every step, tool call, and LLM call.
7. **Fetched content is untrusted data.** Source text is never treated as instructions. Wrap it in clear delimiters, tell the model it is data, and add a test with a source containing "ignore previous instructions".
8. **Every claim in a report cites source IDs and a verbatim evidence quote** that exists in the stored source content. Unsupported claims never appear as findings.
9. **Deterministic mode for tests and evaluation.** LLM and tools both have mock and record/replay implementations. CI never touches the network.
10. **Secrets in environment variables only.** Never commit keys. Provide `.env.example`.
11. **Interfaces first, enforcement later.** Introduce interfaces (tool registry, budget tracker, event emitter) early as thin implementations so later milestones add enforcement without refactoring agents.
12. **Verify third-party APIs against the installed version** before relying on them (LangGraph interrupts and checkpointers, Langfuse SDK, LLM SDK structured output, Next.js, React Flow). Write a short spike script or note per library.
13. Keep it small. No more than the agents listed, one orchestration framework (LangGraph), no extra vector databases.

## 2. Tech stack additions

Backend: FastAPI, Pydantic v2, SQLAlchemy, httpx, trafilatura, jinja2, tenacity or a small retry helper, sse-starlette. Orchestration: LangGraph with a checkpointer (M22). Observability: Langfuse (M18). DB: SQLite default, Postgres via `DATABASE_URL` (M16). Frontend: Next.js, TypeScript, Tailwind, shadcn/ui, React Flow, API types generated from OpenAPI. Testing: pytest, pytest-asyncio, Playwright (smoke, optional).

## 3. Target repository layout

```
research-intelligence-os/
├── configs/   intents, routing, workflows, budgets, tools, models, eval (+ recovery.yaml)
├── data/      train.jsonl validation.jsonl test.jsonl hard_test.jsonl LABELING_GUIDE.md examples/
├── research_os/
│   ├── contracts/  router/ llm/ tools/ agents/ harness/ graph/ memory/ artifacts/ observability/ api/
├── evaluation/   dataset/ cassettes/ runner.py metrics.py judges.py failure_injection.py report_results.py
├── experiments/  router/ ablations/ results/ baseline_results.json
├── frontend/     Next.js app
├── tests/        unit/ integration/ e2e/
├── Makefile      data, train, eval-router, run, test, eval, eval-faults, ui
└── README.md
```

## 4. Core contracts (research_os/contracts/, M8)

RoutingMode(DIRECT|LLM_FALLBACK); RouterOutput(intent, confidence, routing_mode, workflow, entities, latency_ms);
Source(source_id, title, url, source_type, published_at, retrieved_at, content, tool);
Claim(claim_id, claim, source_id, evidence_quote, extractor_confidence, verdict, verdict_rationale);
ResearchStep(id, objective, agent, tools, depends_on, status, attempts, outputs);
ResearchPlan(plan_id, workflow, steps, estimated_cost); ExecutionBudget(max_steps, max_tool_calls, max_searches, max_llm_calls, max_runtime_seconds);
RunEvent(run_id, seq, ts, node, type, payload); RunStatus(PENDING, AWAITING_APPROVAL, RUNNING, SUCCESS, PARTIAL, FAILED, CANCELLED);
FailureType(SEARCH_TIMEOUT, INVALID_SOURCE, MALFORMED_LLM_OUTPUT, NO_USEFUL_SOURCES, CONFLICTING_EVIDENCE, TOOL_FAILURE, AGENT_FAILURE, BUDGET_EXHAUSTED, IMPOSSIBLE_PLAN).
RunContext per run: run_id, config snapshot, BudgetTracker, EventEmitter, Recorder, LLM client, tool registry, source store.

## 5. Milestones (acceptance criteria summarized; see user's original text for full detail)

- **M6.0** Inspect/restructure with `git mv`, docs/MIGRATION.md, `.env.example`, Makefile targets, ARCHITECTURE stub, role-based `configs/models.yaml`. Acceptance: old tests pass, no behavior change.
- **M6.1** Baseline experiment on the healthcare domain: LLM-only/MiniLM-only/Hybrid -> `experiments/{baseline_results.json,confusion_matrix.png,routing_benchmark.csv}`; `make eval-router` reproduces; tag `baseline-m6`.
- **M7** Research intents (FACT_LOOKUP, TECHNOLOGY_COMPARISON, LITERATURE_REVIEW, TECHNICAL_DEEP_DIVE, COMPETITIVE_INTELLIGENCE, MARKET_RESEARCH, RESEARCH_MONITORING, GENERAL_RESEARCH); LABELING_GUIDE first; 100-150 examples/intent JSONL; hard_test ~60; retrain, ablation, recalibrate. Confirm the "latest on X" -> GENERAL_RESEARCH rule with the user.
- **M7.1** RouterOutput + deterministic entity extraction (gazetteer YAML, quoted strings, capitalized tokens, dates, latest/recent); benchmark rerun.
- **M8** Contracts, RunContext, EventEmitter (JSONL sink), Recorder, thin ToolRegistry, tools (web_search provider interface, github, arxiv, pubmed, fetch_url + parser with SSRF protection), fixture + replay modes, stable source IDs.
- **M9** Supervisor + workflows.yaml (DIRECT: zero LLM calls; LLM_FALLBACK: structured call, fallback GENERAL_RESEARCH).
- **M10** Planner (template instantiation, optional bounded LLM refinement, deterministic validation, CostEstimate) + LangGraph executor.
- **M11** Search, Source Analysis, Evidence (verbatim-quote check), Synthesis agents; versioned prompts; untrusted-content delimiters.
- **M12** Verifier: deterministic checks + LLM judge, contradiction detection, gate logic; verifier_gold.jsonl (30 pairs).
- **M13** Full ToolSpec registry: schema validation, timeouts, retries, permissions, policies, direct-import lint test.
- **M14** BudgetTracker + budgets.yaml; exhaustion -> PARTIAL artifact.
- **M15** Recovery policy table (configs/recovery.yaml), capped, `terminate_safely`.
- **M15.5** Run API (POST/GET runs, SSE events with replay from seq, report, health), OpenAPI types, minimal Next.js debug page.
- **M16** SQLAlchemy persistence + Alembic, `memory_search` using MiniLM embeddings, reuse policy, monitoring diff.
- **M17** ResearchReport model; Markdown/JSON/PDF renderers; generated limitations; citation-integrity test.
- **M18** Langfuse tracing (no-op without keys).
- **M19** Evaluation dataset (100-300 tasks), frozen cassettes, 30-task gold subset, no overlap with router data.
- **M20** Four-system evaluation + harness ablations, judges with agreement reporting, results generated.
- **M21** Failure injection scenarios, crash rate 0.
- **M22** Human-in-the-loop approvals via LangGraph interrupts.
- **M23** Frontend: workspace, live execution graph, evidence explorer, report, history; Playwright smoke.
- **M24 (optional)** Voice adapters.

## 6. Cross-cutting

Tests: unit/integration/e2e, CI-safe (no network/keys). Required: budget enforcement, schema validation, citation integrity, prompt-injection resistance, SSRF blocking, record/replay determinism, recovery cap. Replay: `python -m research_os.harness.replay <run_id>`. Docs: ARCHITECTURE.md with Mermaid per milestone; README generated tables. Commits: `M<n>: <title>`, tag `m<n>-done`.

## 7. Definition of done / 8. Order if short / 9. What not to add

See the user's original message: baseline tagged; research router retrained+calibrated; RouterOutput+entities; planner workflow; verifier gate; registry/budgets/recovery; memory; reports MD/JSON/PDF; Langfuse no-op; four-system eval + ablations from replay; failure injection 0 crashes; approvals; five-screen frontend; README from experiments/results. Short-time order: M6.0 → M6.1 → M7 → M7.1 → M8 → M9 → M10 → M11 → M12, then M13-M15.5, then M19-M20, then M16 M17 M18 M21 M22 M23, M24 last. Do not add: extra agents/frameworks, multiple vector DBs, MCP wrappers without need, agent-to-agent chat, voice before text is reliable, cron monitoring.
