# Failure injection

offline: fixture corpus + deterministic demo LLM; D = full harness, C = minimal harness (contrast)

| System | expectations met | crash rate | recovery success | safe termination |
|---|---|---|---|---|
| D | 100% | 0% | 100% (5) | 100% (3) |
| C | 27% | 0% | 0% (5) | 0% (3) |

| Scenario | Failure type | D | C |
|---|---|---|---|
| search_api_timeout | SEARCH_TIMEOUT | met (SUCCESS) | NOT met (FAILED) |
| invalid_urls | INVALID_SOURCE | met (SUCCESS) | met (SUCCESS) |
| no_sources_found | NO_USEFUL_SOURCES | met (PARTIAL) | NOT met (FAILED) |
| contradictory_sources | CONFLICTING_EVIDENCE | met (SUCCESS) | NOT met (SUCCESS) |
| malformed_tool_arguments | TOOL_FAILURE | met (SUCCESS) | met (SUCCESS) |
| llm_unsupported_claim | UNSUPPORTED_CLAIM | met (SUCCESS) | NOT met (SUCCESS) |
| tool_returns_garbage | INVALID_SOURCE/garbage | met (PARTIAL) | NOT met (SUCCESS) |
| impossible_plan | IMPOSSIBLE_PLAN | met (SUCCESS) | NOT met (PARTIAL) |
| budget_exceeded | BUDGET_EXHAUSTED | met (PARTIAL) | NOT met (SUCCESS) |
| prompt_injection_in_source | PROMPT_INJECTION | met (SUCCESS) | met (SUCCESS) |
| llm_malformed_output | MALFORMED_LLM_OUTPUT | met (SUCCESS) | NOT met (PARTIAL) |
