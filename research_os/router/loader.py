"""Resolve a --model argument ('baseline' or a directory) to a router."""
from __future__ import annotations

from pathlib import Path

from research_os.router.baseline_router import BaselineRouter
from research_os.router.model import IntentRouter


def load_router(spec: str, baseline_dir: str = "models/baseline") -> tuple[IntentRouter, str, bool]:
    """Return (router, display_name, fine_tuned)."""
    path = baseline_dir if spec == "baseline" else spec
    if (Path(path) / "head.joblib").exists():
        return BaselineRouter.load(path), "baseline", False
    return IntentRouter.load(path), Path(path).name, True
