# How the harness works

This is a deep-dive into `research_os/harness/` (+ `research_os/tools/registry.py`, which enforces the same rules
for tool calls). Read `docs/CODEBASE_WALKTHROUGH.md` first if you haven't — this assumes you know the overall
request -> router -> supervisor -> planner -> executor -> report flow.

## 1. What problem the harness solves

The agent pipeline (planner + 5 step-agents) is the part that does the actual research. Left on its own, a step
could: call tools forever, fetch an internal/private URL, run out of money on LLM calls, crash on a malformed tool
response, loop retrying the same failing thing, or finish and no one would know why a claim was dropped.

The harness is everything that turns that loose pipeline into something with **limits, a recovery strategy, an
audit trail, and a pause button**. It sits *between* agents and the outside world (tools, LLMs) and *around* the
executor (budgets, recovery). Nothing talks to a tool or an LLM directly — it always goes through the harness.

```
agent code  ->  ToolRegistry.call() / ContextLLM.complete()  ->  budget check  ->  recorder (record/replay)
                                                                ->  real tool / LLM  ->  events emitted
executor    ->  on any failure  ->  RecoveryManager.decide()  ->  retry / alternate / replan / terminate
graph       ->  approval_node()  ->  LangGraph interrupt (pause for a human) before/after execution
```

The six pieces, each in its own file:

| File | Responsibility |
|---|---|
| `tools/registry.py` | the only door through which an agent calls a tool: schema, permission, policy, timeout, retry, output cap |
| `harness/budgets.py` | hard numeric limits, checked *before* incrementing, never after |
| `harness/recovery.py` | failure type -> ordered list of actions, with per-step/per-run caps so it can't loop forever |
| `harness/approvals.py` | thresholds for pausing a run for a human decision (LangGraph interrupt) |
| `harness/replay.py` | records every tool/LLM call; replays them later with zero network/LLM access |
| `harness/events.py` | the append-only event log everything else observes (UI SSE, Langfuse, JSONL file) |
| `harness/verification.py` | deterministic grounding checks + the PASS/FAIL gate (the verifier agent calls into this) |
| `harness/faults.py` | test-only fault injection proxies used by the offline failure-injection evaluation |
| `harness/context.py` | `RunContext` — bundles budget/events/recorder/tools/llm/sources for one run; agents never touch globals |

---

## 2. `RunContext` — the one object every agent receives

`harness/context.py`

Every agent method gets a `ctx: RunContext` with:

* `ctx.budget` — the `BudgetTracker` for this run
* `ctx.events` — the `EventEmitter` for this run
* `ctx.recorder` — the `Recorder` (record / replay / off)
* `ctx.tools` — the `ToolRegistry`
* `ctx.sources` — a `SourceStore` (sources gathered so far, deduplicated by `source_id`; a longer fetch of the
  same page replaces a shorter snippet)
* `ctx.llm` — a `ContextLLM` wrapper (see below)
* `ctx.memory` — the persistent `ResearchStore`, or `None`

This is a deliberate design choice: agents never reach for a global config, a global HTTP client, or a global
counter. Everything flows through one object, which is also exactly what gets threaded into the recorder/replay
machinery and the approval checkpointer.

**`ContextLLM.complete()`** (in the same file) is the LLM-calling path every agent uses instead of calling an LLM
client directly:
1. `ctx.budget.count("llm_calls")` — budget checked *first*.
2. Emits `llm:<model> started`.
3. Goes through `ctx.recorder.call(...)` (record/replay).
4. Emits `llm:<model> finished` with token counts, or `failed` with the error.

So every LLM call, anywhere in the system, is budgeted, recorded, and logged by construction — an agent can't
accidentally bypass this because there's no other way to call an LLM.

---

## 3. The tool registry — the only door to the outside world

`research_os/tools/registry.py`, built by `research_os/tools/__init__.py::build_registry()`

`ToolRegistry.call(name, args, ctx, agent=...)` runs, in this exact order, for **every** tool call:

1. **Tool exists?** Unknown tool name -> `ToolError(TOOL_FAILURE)`.
2. **Permission check** — `configs/tools.yaml: permissions:` maps each agent to the tool permissions it's granted
   (e.g. `search_agent: [research]`, `evidence_agent: []`). A tool call from an agent without the matching
   permission is refused *before* anything runs. This is how, structurally, the evidence/verifier/synthesis
   agents simply cannot call `fetch_url` or `web_search` even if a prompt somehow asked them to — they have
   `permissions: []`.
3. **Schema validation** — each tool has a Pydantic `input_schema` (`SearchInput`, `FetchInput`,
   `ScholarlySearchInput`, `MemorySearchInput` in `tools/schemas.py`). Invalid args -> `ToolError(TOOL_FAILURE)`
   with the validation error, never a Python exception leaking out.
4. **Pre-checks** — tool-specific extra checks run here. `fetch_url` registers `check_fetch_policy`
   (`harness/policies.py`): the target host must not be in `blocked_domains`, and if `allowed_domains` is
   non-empty, the host must be on it. (Empty `allowed_domains` = any public host, which is the current default.)
5. **Budget check** — `ctx.budget.count_all(...)`: every tool call counts against `tool_calls`; search tools
   (`web_search`, `github_search`, `arxiv_search`, `pubmed_search`) *also* count against `searches`.
6. **Execution with timeout** — the tool function runs in a 1-worker thread pool with a timeout (per-tool override
   or `registry.default_timeout_s`, default 15s). A timeout becomes `ToolError(SEARCH_TIMEOUT)`; any other
   exception becomes `ToolError(TOOL_FAILURE)`. Agents never see a raw exception from a tool — only `ToolError`.
7. **Retries** — `configs/tools.yaml: registry: {max_retries: 2, backoff_s: 0.5, retry_on: [SEARCH_TIMEOUT,
   TOOL_FAILURE]}`. Only failure types in `retry_on` are retried, with exponential backoff (`backoff_s * 2^n`).
   Each retry attempt also re-counts against the budget (a retry is not free).
8. **Record/replay** — the whole retry loop above runs *inside* `ctx.recorder.call("tool", name, args, run)`, so a
   replayed run gets the exact recorded result (including which attempt succeeded) without hitting the network.
9. **Output capping** — `max_content_chars` (20,000 by default, `configs/tools.yaml: limits:`) truncates fetched
   source text so one huge page can't blow out context windows or storage.
10. **Events + source store** — `tool:<name> started/retry/finished/failed` events are emitted throughout, and any
    `Source` objects the tool returns are added to `ctx.sources`.

### SSRF protection (`tools/fetch_url.py`)

`fetch_url` is the one tool that reaches arbitrary user-influenced URLs, so it gets extra scrutiny beyond the
generic registry:

* Only `http`/`https` schemes are allowed.
* The hostname is resolved, and **every** resolved address is checked against a private/loopback/link-local/
  multicast/reserved/CGNAT (`100.64.0.0/10`) blocklist (`check_url_safe`). IPv4-mapped IPv6 addresses are unwrapped
  first so they can't sneak past the check.
* **DNS-rebinding protection**: the IP that passed the safety check is the IP actually connected to
  (`PinningTransport`, via an httpx request extension `pin_ip`) — the resolver isn't asked again between the
  check and the connection, closing the classic "DNS answer changes between check and connect" attack window. The
  `Host` header and TLS SNI still use the real hostname, so certificate validation is unaffected.
* Redirects are followed manually (up to `max_redirects`, default 3), and **each hop is re-vetted** through
  `check_url_safe` — a safe URL redirecting to an internal address is still blocked.
* `trust_env=False` on the httpx client — proxy environment variables are ignored, so a compromised environment
  variable can't redirect fetches through an attacker's proxy.
* Response size (`max_bytes`, 2MB default) and content-type (`ALLOWED_TYPES`: html/xhtml/plain/markdown/xml) are
  enforced while streaming, not after downloading the whole thing.

### Prompt-injection hygiene (`agents/safety.py`)

Fetched web content is untrusted data. Before it reaches any prompt, the source agent runs `strip_injection()`,
which removes whole sentences matching instruction-like patterns (`"ignore previous instructions"`, `"you are now
..."`, `"reveal the system prompt"`, `"admin/developer/dan/jailbreak mode"`, etc.). This is explicitly a
first line of defense, not a guarantee — sources are also delimited and labelled as data in every prompt, and
ultimately the verifier still has to find a verbatim quote for any claim, so even content that survives an
injection attempt can't inject an unsupported claim into the report.

---

## 4. Budgets — checked before, not after

`research_os/harness/budgets.py`

`BudgetTracker` keeps four counters (`steps`, `tool_calls`, `searches`, `llm_calls`) plus wall-clock elapsed time,
against an `ExecutionBudget` (one of the profiles in `configs/budgets.yaml`: `small` / `medium` / `large`, chosen
per-workflow via `budget_profile` in `configs/workflows.yaml`):

```yaml
small:  {max_steps: 10, max_tool_calls: 14, max_searches: 4,  max_llm_calls: 12, max_runtime_seconds: 120}
medium: {max_steps: 18, max_tool_calls: 30, max_searches: 8,  max_llm_calls: 24, max_runtime_seconds: 300}
large:  {max_steps: 28, max_tool_calls: 60, max_searches: 16, max_llm_calls: 45, max_runtime_seconds: 600}
```

The core method is `count_all(kinds: dict[str, int])`, and it's written so a check can never be bypassed by a
race: under one lock, it (a) checks elapsed wall-clock time against `max_runtime_seconds`, (b) checks every
requested counter would stay within its limit, and only if **everything** passes does it increment anything. A
refusal raises `BudgetExhausted` (carrying `kind`, `limit`, `used`) and increments nothing — so a refused call
never silently "spends" part of its cost.

`BudgetExhausted` is not an ordinary exception: it carries `failure_type = FailureType.BUDGET_EXHAUSTED`, so when
it's thrown from inside a step (via a tool call or an LLM call), the executor's `classify_failure()` picks it up
automatically and recovery (see below) always treats it as a hard stop, regardless of what the recovery policy
file says for other failure types.

`ctx.budget.snapshot()` (counters + limits) is what shows up in the UI's budget meters and in every report's
metadata.

---

## 5. Recovery — classify, look up, apply, bounded

`research_os/harness/recovery.py` + `configs/recovery.yaml`

This is the part that makes a tool failure or a bad LLM output *not* a crash.

### Failure types

```python
SEARCH_TIMEOUT, INVALID_SOURCE, MALFORMED_LLM_OUTPUT, NO_USEFUL_SOURCES,
CONFLICTING_EVIDENCE, TOOL_FAILURE, AGENT_FAILURE, BUDGET_EXHAUSTED, IMPOSSIBLE_PLAN
```

Every exception that escapes a step is mapped to exactly one of these by `graph/executor.py::classify_failure()`:
`ToolError` carries its own `failure_type`; `MalformedLLMOutput` (schema validation failed after retries) maps to
`MALFORMED_LLM_OUTPUT`; anything else with a `.failure_type` attribute uses that (this is how `BudgetExhausted`
flows through); everything else falls back to `AGENT_FAILURE`.

### The policy table

`configs/recovery.yaml` maps each failure type to an **ordered list of actions**:

```yaml
limits: {max_per_step: 4, max_per_run: 12, max_replans: 1}
alternates:
  web_search: [github_search, arxiv_search]
  github_search: [web_search]
  arxiv_search: [pubmed_search, web_search]
  pubmed_search: [arxiv_search, web_search]
  fetch_url: [web_search]
policy:
  SEARCH_TIMEOUT:       [retry, alternate_tool, terminate_step]
  INVALID_SOURCE:       [skip_source, alternate_tool, terminate_step]
  MALFORMED_LLM_OUTPUT: [retry_with_error, terminate_step]
  NO_USEFUL_SOURCES:    [broaden_query, alternate_tool, replan, ask_user]
  CONFLICTING_EVIDENCE: [proceed]
  TOOL_FAILURE:         [retry, alternate_tool, terminate_step]
  AGENT_FAILURE:        [retry, skip_step]
  BUDGET_EXHAUSTED:     [terminate_safely]
  IMPOSSIBLE_PLAN:      [replan, terminate_safely]
```

**The n-th failure of a given step takes the n-th action in its list.** `RecoveryManager.decide()`:

1. If the failure is `BUDGET_EXHAUSTED` -> always `terminate_safely`, no matter what the policy file says (a
   spent budget is a hard stop by design, not a policy choice).
2. If the **per-run** cap (`max_per_run: 12`) is reached -> `terminate_safely` ("too many recoveries overall").
3. If the **per-step** cap (`max_per_step: 4`) is reached for this step -> `terminate_step`.
4. Otherwise walk the policy list for this failure type starting from where this step left off last time,
   skipping any action that isn't *applicable* right now (`_applicable()`):
   - `alternate_tool` only applies if there's an untried alternate left (from the `alternates:` map above).
   - `broaden_query` only applies to `search_agent` steps.
   - `skip_source` only applies to `search_agent`/`source_agent` steps.
   - `replan` only applies if `max_replans` (1) hasn't been used yet.
   - Everything else is always applicable.
   If nothing in the remaining list is applicable, the step is terminated.

Every decision emits a `recovery` event (step, failure type, action, attempt, reason) — this is what you see in
the UI's event log and what `evaluation/failure_injection.py` checks against.

### Applying a decision (`RecoveryManager.apply`)

Before the step is re-run, its `params` are mutated so the retry is actually different from the failed attempt:

| Action | What changes |
|---|---|
| `retry_with_error` | `params["last_error"]` is set, so the agent/prompt can see what went wrong |
| `broaden_query` | `params["broaden_level"]` incremented (search agent widens its query) |
| `alternate_tool` | the step's `tools` list is swapped to the next untried alternate; `tools_tried` tracks what's been used so it's never retried twice |
| `skip_source` | the failing source id is added to `params["skip_source_ids"]` so it's excluded next attempt |

### What the executor does with the decision (`graph/executor.py`)

Each step runs in a loop: on success it's marked `succeeded` and the loop breaks. On failure, `classify_failure()`
runs, a `recovery.decide()` call happens, and then:

* **Re-run actions** (`retry`, `retry_with_error`, `broaden_query`, `alternate_tool`, `skip_source`) ->
  `recovery.apply()` mutates the step, loop continues (this re-run also re-increments `ctx.budget.count("steps")`
  — a retry is a step too).
* **`skip_step`** -> the step (and anything depending on it) is marked `skipped`.
* **`proceed`** on a verifier `GateFailed` -> the step is force-accepted as `succeeded` anyway (used for
  `CONFLICTING_EVIDENCE`: a conflict is surfaced in the report, not treated as fatal).
* **`terminate_safely`** -> the whole run halts; `res.halted_reason` is set.
* **`ask_user`** -> the run halts with a canned `ask_user` message ("I could not find useful, verifiable
  sources... please rephrase").
* **`replan`** -> `res.replan_requested = True`; the graph's `replanner_node` calls `planner.replan()` with the
  failure reason and re-executes the *new* plan (history from the first attempt is preserved via
  `merge_history()`).
* **`terminate_step`** (anything else) -> the step is marked `failed`; dependents are skipped since their
  dependency didn't succeed.

Steps whose dependencies didn't succeed are never run — `_topological()` orders steps by `depends_on`, and the
executor explicitly checks `status_of[d] != "succeeded"` before running a step.

Final run status: `SUCCESS` if every step succeeded, `PARTIAL` if some did (or the run was deliberately halted),
`FAILED` only if literally nothing succeeded. **A `PARTIAL` artifact is always producible** — `build_partial_artifact`
/ `empty_partial` in `recovery.py` build a valid response (what completed, what didn't, which claims survived,
budget used) even for a run that produced zero claims. Nothing in this system "throws a 500 and gives up."

---

## 6. The verifier's deterministic gate (recovery's other input)

`research_os/harness/verification.py` (invoked from `agents/verifier.py`)

This is the harness's other failure producer, separate from tools: a step can fail not because a tool errored,
but because the **verifier rejected too many claims**. This is `research_os/graph/executor.py::GateFailed`.

For each claim, `deterministic_checks()` runs *before* any LLM judging, and rejects a claim if:
* its `source_id` doesn't point at a source actually in `ctx.sources`;
* its `evidence_quote` is not a verbatim substring of that source's stored text (`find_quote`);
* the source's type isn't in the workflow's `allowed_source_types`;
* the workflow requires dated sources (`require_dates`) and the source has no `published_at`;
* the claim uses comparative/superlative language (`faster`, `better`, `most`...) but the quote doesn't contain a
  comparative word *or* a number;
* the claim states a number that doesn't appear in the quote;
* the claim uses universal language (`always`, `never`, `all`, `every`) not present in the quote;
* the claim uses causal language (`because`, `causes`, `leads to`...) not present in the quote.

Only claims that pass *all* of these go on to the LLM judge for entailment — and the judge can only downgrade a
claim further, never rescue one the deterministic checks already rejected.

`detect_conflicts()` groups claims about the same entity+attribute that disagree in polarity (one asserts, one
negates) **across two different sources** (same-source "conflicts" are explicitly excluded, since that was a past
bug). Detected conflicts are then confirmed by one extra LLM call (`prompts/conflict_confirm.v2.md`) before being
treated as a real conflict.

`gate()` turns verdict counts into PASS/FAIL using the workflow's thresholds (`min_supported_claims`,
`max_unsupported_rate`, `required_source_types` from `configs/workflows.yaml`). A confirmed conflict counts toward
the "supported" minimum instead of failing the run — conflicts are surfaced in the report, not fatal. A FAIL
becomes `GateFailed`, which the executor classifies as `FailureType.NO_USEFUL_SOURCES` and hands to recovery like
any other failure (so `[broaden_query, alternate_tool, replan, ask_user]` applies to it too).

---

## 7. Record / replay — determinism as a first-class feature

`research_os/harness/replay.py`

Every tool call and every LLM call passes through `ctx.recorder.call(kind, name, args, fn)`. The `Recorder` has
three modes:

* **`off`** — just calls `fn()`. Normal live operation.
* **`record`** — calls `fn()`, and stores `{kind, name, key, args, ok, result|error}` in memory; `save()` writes
  it all as JSONL (one line per call) to `runs/<run_id>/recording.jsonl`.
* **`replay`** — never calls `fn()`. Looks up a queue of recorded entries keyed by `call_key(kind, name, args)` —
  a SHA-256 hash of the canonical JSON of `[kind, name, args]` — and pops the next matching entry. If nothing
  matches, it's a `ReplayMiss` (recorded as a miss, not silently invented data). If the recorded entry failed, the
  original error (`ToolError` or a generic `RecordedError`) is faithfully re-raised.

Because the key is a hash of the exact arguments, and entries for the same key are served in order (a `deque`),
replaying a run reproduces not just *what* was called but *in what order with what result*, including which retry
attempt in a backoff loop eventually succeeded.

`python -m research_os.harness.replay <run_id>` is the CLI: it rebuilds a `RunService` whose LLM is `_Forbidden`
(any `.complete()` call is a bug — everything must come from the recording) and whose tools replay from the
recording, re-runs the pipeline, and diffs the resulting report against the original (`normalize_report()` strips
wall-clock fields like `generated_at` and `elapsed_seconds` first, since those are explicitly excluded from the
determinism guarantee). Exit code 0 = identical, 1 = different (with a list of dotted JSON paths that differ).

This is also exactly how the **evaluation harness** gets repeatable, free, offline runs across all 120 tasks: real
web/LLM responses are recorded once (`make eval-record`, costs API calls, writes `evaluation/cassettes/`), then
`make eval` replays them indefinitely.

---

## 8. Approvals — pausing for a human

`research_os/harness/approvals.py`, wired into `research_os/graph/workflow.py`

`ApprovalPolicy` (from `configs/budgets.yaml: approvals:`) holds thresholds:

```yaml
approvals:
  max_searches: 6
  max_sources: 30
  max_llm_calls: 12
  publish: false
  expiry_seconds: 900
```

`needs_plan_approval(cost, policy)` compares the planner's `CostEstimate` against these thresholds and returns a
list of human-readable reasons (e.g. `"8 searches (threshold 6)"`). An empty list means the plan runs without
asking. A value *at* the threshold does not trigger approval — only strictly above it does.

In the graph (`workflow.py::approval_node`), if approval is needed, the node calls LangGraph's `interrupt(...)`
with the plan's cost estimate, current budget limits, and the reasons — this **pauses the whole graph**, not just
one node, and control returns to the API layer, which surfaces it to the UI as a pending approval. When a human
approves or rejects (via the API), the graph resumes exactly where it paused. Rejecting (or letting it expire,
`expiry_seconds: 900`) sets `status = CANCELLED` with the rejection reason.

There's a second interrupt point, `publish_node`, gated by `approvals.publish` (`false` by default) — if enabled,
even a *finished* report pauses for a "approve before this goes out" decision.

Because LangGraph needs to serialize the paused state across the interrupt (it might resume in a different
process/request), `make_checkpointer()` builds an `InMemorySaver` with an explicit allow-list of the custom types
(`ResearchPlan`, `ResearchStep`, `CostEstimate`, `RunStatus`, `Source`, `Claim`, `RouterOutput`, `RoutingMode`,
`SupervisorDecision`, `ExecutionResult`) that are allowed to cross that boundary — an intentional whitelist rather
than "serialize anything."

---

## 9. Events — the thing everything else is built on

`research_os/harness/events.py`

`EventEmitter` is a thread-safe, append-only, monotonically-sequenced log of `RunEvent(run_id, seq, ts, node,
type, payload)` for one run. Every other harness piece (and every agent) calls `ctx.events.emit(node, type,
payload)` — there's no other channel.

* Kept in memory (`events.events`) *and* appended to `runs/<run_id>/events.jsonl` if a `runs_dir` was given — the
  JSONL file is explicitly called out as "the local source of truth for observability" in the module docstring.
* `add_listener(fn)` lets other systems subscribe synchronously — this is how:
  - the **SSE endpoint** streams live events to the frontend, and can **resume** a dropped connection via
    `since(seq)` (replay only events with `seq >= last_seen`);
  - the **Langfuse tracer** (`observability/tracing.py`) turns the event stream into nested spans (router /
    supervisor / planner / step / tool / LLM generation), with recovery/budget/retry events surfaced as
    `WARNING`-level trace events and failures as `ERROR` — entirely additively; a tracing exception is always
    caught and can never fail the run, and it's a no-op if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't set.

---

## 10. Fault injection — testing the harness itself

`research_os/harness/faults.py` (used only by `evaluation/failure_injection.py`, never in production)

Two proxies that wrap the real registry/LLM and misbehave on command, so the harness's recovery behavior can be
tested deterministically and offline:

* **`FaultyRegistry`** wraps a `ToolRegistry`. Rules can be registered per tool: `fail()` (raise a given
  `ToolError` for the first N calls), `empty()` (return no sources), `replace()` (return a canned `ToolResult`),
  `tamper_args()` (corrupt the call arguments). Everything not matched by a rule passes straight through to the
  real registry.
* **`FaultyLLM`** wraps an LLM client. `malformed()` returns truncated/invalid JSON for prompts matching a prefix
  (triggers `MALFORMED_LLM_OUTPUT`); `add_overreaching_claim()` injects a claim into evidence-extraction output
  that goes beyond its quote (tests that the verifier's deterministic checks catch it, not just the LLM judge).

This is the machinery behind the README's "Failure injection" table — 11 scenarios (timeout, invalid URLs, no
sources found, contradictory sources, malformed tool args, unsupported claim, garbage tool output, impossible
plan, budget exhausted, prompt injection, malformed LLM output), each checking that the full harness (`D`) reaches
the expected outcome and a minimal harness (`C`, no verifier/recovery) does not.

---

## 11. Putting it together: one step, fully annotated

```
executor_node calls execute_plan(plan, agents, ctx, request, recovery=RecoveryManager())
  for step in topological order:
    if step.depends_on didn't all succeed -> skip step, continue
    loop:
      step.attempts += 1
      ctx.budget.count("steps")                      # <- budget check #1 (before running)
      agent.run(step, inputs, ctx, request)           # <- agent does its job
        -> may call ctx.tools.call(...)               # <- permission, schema, policy, budget #2,
                                                        #    timeout, retry, record/replay, event, source store
        -> may call ctx.llm.complete(...)              # <- budget #3, record/replay, event
        -> (verifier step) calls verification.gate()   # <- may raise GateFailed instead of returning normally
      if succeeded: mark step succeeded, break
      else:
        ft = classify_failure(exc)                     # map exception -> FailureType
        d = recovery.decide(step, ft, ctx, exc)         # policy table + per-step/per-run caps
        emit "recovery" event
        if d.action in {retry, retry_with_error, broaden_query, alternate_tool, skip_source}:
          recovery.apply(d, step, exc)                  # mutate step.params / step.tools
          continue loop                                  # re-run the SAME step, budget #1 charged again
        else:
          mark step failed/skipped/succeeded(proceed), possibly halt the whole run
          break
  compute RunStatus (SUCCESS / PARTIAL / FAILED) from which steps succeeded
```

Every one of these layers (permission check, schema check, policy check, budget check, timeout, retry, recovery
action, approval gate) is independently testable and independently visible in the event log — that's the point of
calling it a "harness" rather than just error handling glued onto the agents.

---

## 12. Where to look next

* `tests/unit/test_budgets.py`, `test_recovery.py`, `test_approvals.py`, `test_faults.py`, `test_harness_core.py`
  — unit tests for exactly the behavior described above, good for seeing edge cases (e.g. what happens at exactly
  the budget limit).
* `tests/unit/test_failure_scenarios.py`, `tests/unit/test_recovery_graph.py` — recovery wired into the full graph.
* `evaluation/failure_injection.py` + `experiments/results/failure_injection.md` — the 11-scenario suite and its
  results for systems C vs D.
* `configs/recovery.yaml`, `configs/budgets.yaml`, `configs/tools.yaml` — the knobs, in one place each.
