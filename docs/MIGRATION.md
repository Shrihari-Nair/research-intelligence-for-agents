# Migration map (old layout -> research_os layout)

Done with `git mv` in M6.0; no behaviour change (109 pre-existing tests still pass, healthcare demo output unchanged).

| Old path | New path | Note |
|---|---|---|
| `src/common.py` | `research_os/common.py` | |
| `src/data/*` | `research_os/router/data/*` | healthcare data tooling, reused in M7 |
| `src/training/*` | `research_os/router/training/*` | SetFit baseline / fine-tune / ablation |
| `src/models/router.py` | `research_os/router/model.py` | `IntentRouter` |
| `src/models/{baseline_router,loader,calibration}.py` | `research_os/router/*.py` | |
| `src/evaluation/*` | `research_os/router/evaluation/*` | router-level eval + healthcare comparison |
| `src/systems/*` | `research_os/router/systems/*` | LLM-only / MiniLM-only / Hybrid routing |
| `src/llm/*` | `research_os/llm/*` | + new `models.py` (role-based model names) |
| `src/{agents,graph,api,voice,factory.py}`, `demo.py` | `research_os/legacy/*` | healthcare pipeline; replaced in M10 / M15.5, kept only as fixtures/tests |
| `tests/test_*.py` | `tests/unit/test_*.py` | |

Not moved yet (moved when the research domain lands in M7): `configs/{intents,training,routing}.yaml`, `data/`, `models/`, `results/`.
Healthcare artifacts stay available for the M6.1 baseline.

## Incident worth knowing

`.gitignore` had `models/`, which also ignored `src/models/` (four source files were never committed in the earlier
branch). The ignore rule is now anchored (`/models/`). During the move an `rm -rf src` removed those four untracked
files; they were recreated from the session history and are covered by the existing tests (router, calibration,
routing, API) which all pass.

## Hardcoded model names

The Gemini model name used to live in `configs/routing.yaml`. It now lives in `configs/models.yaml` under
role names (`router_fallback`, `supervisor`, `planner`, `evidence`, `verifier_judge`, `synthesis`, `eval_judge`);
code calls `research_os.llm.models.model_for_role(role)`.
