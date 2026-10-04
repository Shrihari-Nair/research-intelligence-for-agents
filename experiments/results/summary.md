# Evaluation summary (2026-10-03)

## How to read this

- 120 tasks, one run per (system, task), **replayed** from frozen sources recorded live; LLM answers come from a content-addressed cache, so reruns are identical. Latency columns are wall-clock/recorded and not part of the determinism guarantee.
- Quality metrics are judged by `gemini-3.5-flash` (a different model from the pipeline's verifier). **Judge agreement vs hand labels: n=60, accuracy=0.85, kappa=0.33.**
- Tasks, intent labels and claim labels were written by the project author. Rates show mean [95% bootstrap CI over tasks]. One live recording, one replay: no run-to-run variance is measured.

## Router

| System | workflow accuracy | macro F1 | routing P50 | routing P95 | routing P99 |
|---|---|---|---|---|---|
| A | 93.3% | 93.3% | 1339.6 ms | 1490.3 ms | 1938.6 ms |
| B | 85.8% | 86.0% | 329.1 ms | 479.2 ms | 517.4 ms |
| C | 88.3% | 88.4% | 421.1 ms | 1722.5 ms | 1981.2 ms |
| D | 88.3% | 88.4% | 308.4 ms | 1425.4 ms | 1825.9 ms |

## Research quality

| System | task completion | end-to-end success | unsupported-claim rate | citation correctness | source relevance |
|---|---|---|---|---|---|
| A | 71.7% [63.3%, 80.0%] | 67.5% [59.2%, 76.7%] | 6.0% [3.9%, 8.3%] | 94.0% [91.7%, 96.1%] | 58.7% [52.6%, 64.2%] |
| B | 71.7% [63.3%, 80.0%] | 67.5% [59.2%, 76.7%] | 6.0% [3.9%, 8.6%] | 94.0% [91.4%, 96.1%] | 59.5% [53.6%, 65.0%] |
| C | 71.7% [63.3%, 80.0%] | 67.5% [59.2%, 76.7%] | 6.1% [3.9%, 8.6%] | 93.9% [91.4%, 96.1%] | 58.8% [52.9%, 64.4%] |
| D | 65.8% [57.5%, 74.2%] | 65.0% [56.7%, 73.3%] | 2.4% [1.2%, 3.8%] | 97.6% [96.2%, 98.8%] | 58.0% [52.4%, 63.5%] |

## Agent system

| System | tool selection | plan success | recovery success | end-to-end |
|---|---|---|---|---|
| A | 98.4% [96.0%, 99.9%] | 100.0% [100.0%, 100.0%] | n/a | 67.5% [59.2%, 76.7%] |
| B | 98.5% [97.1%, 99.6%] | 100.0% [100.0%, 100.0%] | n/a | 67.5% [59.2%, 76.7%] |
| C | 99.3% [98.4%, 100.0%] | 100.0% [100.0%, 100.0%] | n/a | 67.5% [59.2%, 76.7%] |
| D | 99.3% [98.4%, 100.0%] | 76.7% [69.1%, 84.2%] | 3.4% [0.0%, 10.3%] | 65.0% [56.7%, 73.3%] |

## Efficiency

| System | LLM calls/run | input tok/run | output tok/run | cost/run (USD) | total cost (USD) | tool calls/run | wall P50 | wall P95 |
|---|---|---|---|---|---|---|---|---|
| A | 3.17 [3.10, 3.26] | 2551 | 1393 | $0.00425 | $0.510 | 3.33 [2.91, 3.75] | 0.7 s | 4.4 s |
| B | 2.14 [2.08, 2.21] | 2111 | 1337 | $0.00397 | $0.477 | 3.17 [2.77, 3.58] | 0.7 s | 0.9 s |
| C | 2.22 [2.13, 2.31] | 2154 | 1354 | $0.00403 | $0.484 | 3.17 [2.77, 3.59] | 0.7 s | 1.0 s |
| D | 3.67 [3.47, 3.89] | 3460 | 1887 | $0.00576 | $0.691 | 3.83 [3.30, 4.42] | 1.0 s | 1.6 s |

## Harness ablations

| System | completion | e2e success | unsupported rate | citation correctness | delta vs D |
|---|---|---|---|---|---|
| D (full harness) | 65.8% [57.5%, 74.2%] | 65.0% [56.7%, 73.3%] | 2.4% [1.2%, 3.8%] | 97.6% [96.2%, 98.8%] | - |
| D-no-verifier | 71.7% [63.3%, 80.0%] | 67.5% [59.2%, 76.7%] | 6.1% [3.9%, 8.6%] | 93.9% [91.4%, 96.1%] | completion +5.8pp, e2e_success +2.5pp, unsupported_rate +3.7pp, citation_correctness -3.7pp |
| D-no-recovery | 65.0% [56.7%, 73.3%] | 64.2% [55.8%, 72.5%] | 2.6% [1.4%, 4.0%] | 97.4% [96.0%, 98.6%] | completion -0.8pp, e2e_success -0.8pp, unsupported_rate +0.2pp, citation_correctness -0.2pp |
| D-no-budgets | 65.8% [57.5%, 74.2%] | 65.0% [56.7%, 73.3%] | 2.6% [1.2%, 4.0%] | 97.4% [96.0%, 98.8%] | completion +0.0pp, e2e_success +0.0pp, unsupported_rate +0.2pp, citation_correctness -0.2pp |

## By task kind

| System | completion | e2e success | unsupported rate |
|---|---|---|---|
| A / normal | 68.1% | 66.7% | 5.2% |
| A / ambiguous | 75.0% | 66.7% | 6.9% |
| A / tempting | 79.2% | 70.8% | 7.3% |
| B / normal | 68.1% | 65.3% | 5.5% |
| B / ambiguous | 75.0% | 70.8% | 6.9% |
| B / tempting | 79.2% | 70.8% | 6.7% |
| C / normal | 68.1% | 65.3% | 5.7% |
| C / ambiguous | 75.0% | 70.8% | 6.7% |
| C / tempting | 79.2% | 70.8% | 6.7% |
| D / normal | 65.3% | 65.3% | 2.1% |
| D / ambiguous | 75.0% | 75.0% | 2.7% |
| D / tempting | 58.3% | 54.2% | 2.8% |
