# Observability (Langfuse)

Tracing is **additive**: the local event log (`runs/<run_id>/events.jsonl`, also streamed over SSE) stays the source of truth.
Tracing is a **no-op unless both keys are set**, and a tracing failure can never fail a run.

## Setup

Langfuse Cloud or self-hosted, same SDK (`langfuse` 4.x, observation API: `start_observation`, `update`, `end`, `create_event`, `score_trace`).

```bash
# .env  (never commit real keys)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com      # or your self-hosted URL; leave unset to use the SDK default
```

`configs/tracing.yaml`: `environment`, `capture_content` (default **false**: prompts and tool arguments are NOT sent, only metadata).

## What is recorded

One trace per run (`research_run`):

* spans: `router`, `supervisor`, `planner` (and `recall`)
* one `agent` observation per plan step (`step:<id>`; the verifier step is an `evaluator`), re-runs after recovery appear as separate spans
* nested `tool:<name>` observations (agent, args only if `capture_content`) and `llm:<model>` **generations** with prompt id/version
  (`evidence.v1`, ...), token usage and latency (observation start/end time)
* events: `recovery:<action>`, `budget:*`, `retry:*`, `info:*`; failures set `level=ERROR` with the failure type
* scores on the trace: `supported_claims`, `unsupported_claim_rate`, `conflicts`, `sources`, `steps_used`, `llm_calls`, `tool_calls`,
  `recoveries`, `run_success`

`GET /runs/{id}` returns `trace_url` when tracing is on.

## Expected span tree

Recorded from a real graph run using an in-memory client that mirrors the SDK signatures (see `tests/unit/test_tracing.py`).
**This is not a screenshot of a live Langfuse project**; a live check needs your keys (see "Verification status").

```text
- research_run [agent]
  * event info:budget
  - router [chain]
  - supervisor [chain]
  - planner [chain]
  - step:search [agent]
    - tool:web_search [tool]
  - step:read [agent]
  - step:extract [agent]
    - llm:default [generation]  usage={'input': 1, 'output': 1}  model=default
  - step:verify [evaluator]
  - step:synthesize [agent]

scores: {'supported_claims': 1.0, 'unsupported_claim_rate': 0.5, 'conflicts': 0.0, 'sources': 4.0, 'steps_used': 5.0, 'llm_calls': 1.0, 'tool_calls': 1.0, 'recoveries': 0.0, 'run_success': 1.0}
```

## Verification status

* Unit tests assert the span tree, nesting, usage, ERROR levels, recovery events, scores and the no-op/failure-safety guarantees.
* A smoke test drives the **real installed SDK (4.16.0)** with an unreachable host: no exception, no hang.
* **Verified against a live self-hosted Langfuse 4.49.0** (own stack: `docker/langfuse/up.sh`, UI on :3100; SDK 4.16.0), with the API run in
  live mode (`RESEARCH_OS_TOOLS=live`: real Gemini, Tavily, GitHub and page fetches). Run "Compare LangGraph and CrewAI for multi-agent workflows"
  (status SUCCESS) produced one trace with 28 observations (8 agent steps, 9 tools, 5 `llm:gemini-2.5-flash` generations with token usage and cost,
  1 evaluator, 3 chains) and 9 trace scores (`supported_claims=15`, `unsupported_claim_rate=0.0625`, `sources=17`, ...).
  `GET /runs/{id}` returned a working `trace_url`.
  (An earlier first check ran with the default fixture tools, not live ones, and is superseded by this one.)
* The live check found one bug the unit tests missed: `trace_url` was empty because `get_trace_url()` needs the root's `trace_id` once the root span has ended
  (fixed, with a test).
* Langfuse 4 "events_only" mode has no `/traces` or `/scores` REST endpoints; observations are read via `/api/public/v2/observations?traceId=...`.
* Not checked: Langfuse Cloud (only the self-hosted build) and `capture_content: true`.
