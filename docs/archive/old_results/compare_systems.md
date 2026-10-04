LLM backend used for hybrid fallback: **gemini**.

### test

| Metric | LLM-only | MiniLM-only | Hybrid |
|---|---|---|---|
| Intent accuracy | 98.5% | 99.2% | 99.2% |
| Macro F1 | 98.5% | 99.2% | 99.2% |
| EMERGENCY recall | 100.0% | 100.0% | 100.0% |
| P50 latency | 1137.9 ms | 7.9 ms | 11.0 ms |
| P95 latency | 1306.8 ms | 11.9 ms | 12.5 ms |
| CPU time / request | 5.5 ms | 35.5 ms | 44.5 ms |
| Cost / request | $0.000125 | $0.000000 | $0.000000 |
| LLM calls | 132 | 0 | 0 |

### hard_test

| Metric | LLM-only | MiniLM-only | Hybrid |
|---|---|---|---|
| Intent accuracy | 92.9% | 80.4% | 87.5% |
| Macro F1 | 92.0% | 77.9% | 86.4% |
| EMERGENCY recall | 100.0% | 87.5% | 87.5% |
| P50 latency | 1212.0 ms | 12.1 ms | 12.5 ms |
| P95 latency | 1447.3 ms | 13.7 ms | 1138.6 ms |
| CPU time / request | 4.8 ms | 49.4 ms | 55.4 ms |
| Cost / request | $0.000128 | $0.000000 | $0.000014 |
| LLM calls | 56 | 0 | 6 |

### End-to-end task success

| | LLM-only | MiniLM-only | Hybrid |
|---|---|---|---|
| Task success | 100.0% | 92.5% | 92.5% |
