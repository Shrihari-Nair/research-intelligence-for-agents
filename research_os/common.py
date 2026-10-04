"""Shared helpers: config loading, seeding, hardware info, JSON output."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logging once for CLI scripts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_intents(path: str = "configs/intents.yaml") -> list[str]:
    """Return the ordered intent label list."""
    return list(load_config(path)["intents"])


def intents_from_cfg(cfg: dict) -> list[str]:
    """Intent list for the domain a training config points at (default configs/intents.yaml)."""
    return load_intents(cfg.get("paths", {}).get("intents", "configs/intents.yaml"))


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def get_hardware_info() -> dict:
    """CPU model, cores, RAM, torch version and torch thread count."""
    return {
        "cpu_model": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }


def write_json(path: str | Path, obj: dict) -> None:
    """Write obj as pretty JSON, creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    log.info("wrote %s", p)


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ without overriding existing vars."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))
