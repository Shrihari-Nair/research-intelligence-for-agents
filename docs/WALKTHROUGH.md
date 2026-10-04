# One request, end to end

A real run, taken from a live execution (Gemini + Tavily/GitHub/page fetches, 2026-10-04), for the request
**"What license does LangGraph use?"**. The numbers below are copied from that run's API output, not typed from memory.

## 1. Router (about 34 ms, no LLM)

The fine-tuned MiniLM classifies the request:

```
intent: FACT_LOOKUP   confidence: 0.956   routing_mode: DIRECT   latency: 33.7 ms
entities: technologies = ["LangGraph"]   (found by the deterministic gazetteer + regex extractor)
```

Confidence is above the 0.60 threshold chosen on validation data, so the supervisor takes the **DIRECT** path: the workflow `fact_lookup` is selected
with **zero LLM calls**. (Below the threshold the supervisor makes one schema-validated Gemini call to pick a workflow, falling back to
`general_research` if the answer is invalid.)

## 2. Plan

`configs/workflows.yaml` instantiates the template into five steps, each with only the tools it may use:

| Step | Agent | Tools | What it does |
|---|---|---|---|
| search | search agent | web_search, github_search | builds queries, ranks and de-duplicates sources |
| read | source agent | fetch_url | upgrades snippets to full pages, picks the best passages, strips prompt-injection sentences |
| extract | evidence agent | none | one Gemini call turns passages into atomic claims, each with a quote that must exist verbatim in the stored source |
| verify | verifier | none | deterministic checks, then an LLM judge, then conflict detection |
| synthesize | synthesis agent | none | writes findings from SUPPORTED claims only; every finding cites claim ids |

The plan is validated (no cycles, known tools, within budget) and gets a cost estimate. Expensive plans pause for human approval; this one did not.

## 3. Execution under the harness

* Budget actually used: 5 steps (limit 10), 5 tool calls (14), 2 searches (4), **4 LLM calls (12)**, 20 s (limit 120 s).
* Every tool call goes through the registry (schema check, permission, domain policy, SSRF check, timeout). Fetched web text is treated as
  untrusted data.
* The run was recorded (every tool and LLM call) so it can be replayed without network access: `python -m research_os.harness.replay <run_id>`.

## 4. Verification

The evidence agent extracted **8 claims** from 10 sources. The verifier ended with **7 SUPPORTED and 1 UNSUPPORTED**:

* SUPPORTED: "LangGraph is an MIT-licensed open-source library that is free to use." Quote: *"Yes. LangGraph is an MIT-licensed open-source library
  and is free to use."*
* SUPPORTED: "The LangGraph Server is a commercial product that requires a valid license key for deployment and operation in production."
* UNSUPPORTED (dropped): a claim whose wording went beyond what its quote says.

Candidate contradictions between sources are confirmed by one extra LLM call; here none were confirmed. (An earlier version flagged these two licence
claims as a conflict and answered with an unrelated claim; that bug is why the confirmation step exists.)

## 5. Report

Executive summary, exactly as produced:

> LangGraph is an MIT-licensed open-source library that is free to use. The LangGraph Server, however, is a commercial product requiring a valid
> license key for production deployment and operation. The [mwaeckerlin/langgraph-agent] image is a self-hosted alternative to
> langchain/langgraph-api for local and on-prem deployments.

Available as JSON, Markdown and PDF (`GET /runs/{id}/report?format=pdf`). The Evidence tab in the UI shows each claim next to its quote highlighted
inside the stored source text.

## 6. What this run shows, and what it does not

* It shows the intended behaviour: a confident router, a bounded plan, claims tied to verbatim quotes, and a nuanced answer (library vs server).
* It is **one run**. Live web results vary from run to run, and the evaluation numbers in the README come from a separate frozen corpus of 120 tasks.
