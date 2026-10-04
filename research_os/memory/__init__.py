"""Persistent research memory."""
from research_os.common import load_config


def memory_cfg(path: str = "configs/memory.yaml") -> dict:
    """Settings from configs/memory.yaml."""
    return load_config(path)
