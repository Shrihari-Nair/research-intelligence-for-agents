"""Latency / resource benchmark: batch-size-1 inference on CPU."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from research_os.common import get_hardware_info, load_config, setup_logging, write_json
from research_os.router.data.load import load_split
from research_os.router.evaluation.banner import format_banner
from research_os.router.loader import load_router

log = logging.getLogger(__name__)


def dir_size_mb(path: str) -> float:
    """Total size of files under a directory in MB (symlinks resolved, e.g. HF cache snapshots)."""
    return sum(f.resolve().stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def model_size_mb(spec: str, path: str, base_model: str) -> float:
    """On-disk size; the baseline also counts its frozen encoder from the HF cache."""
    size = dir_size_mb(path)
    if spec == "baseline":
        from huggingface_hub import snapshot_download
        size += dir_size_mb(snapshot_download(base_model, local_files_only=True))
    return size


def percentiles(ms: list[float]) -> dict:
    """P50/P95/P99/mean and throughput from per-request latencies (ms)."""
    a = np.asarray(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()),
            "throughput_rps": float(1000.0 / a.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    b = cfg["benchmark"]
    threads = args.threads or b["threads"]
    torch.set_num_threads(threads)
    log.info("torch threads=%d", threads)

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    path = cfg["paths"]["baseline_model"] if args.model == "baseline" else args.model
    router, name, tuned = load_router(args.model, cfg["paths"]["baseline_model"])
    rss_mb = proc.memory_info().rss / 1e6
    texts = load_split("test", cfg["paths"]["processed"])["text"].tolist()
    for i in range(b["warmup"]):
        router.predict(texts[i % len(texts)])
    lat = []
    for i in range(b["runs"]):
        t0 = time.perf_counter()
        router.predict(texts[i % len(texts)])
        lat.append((time.perf_counter() - t0) * 1000)
    out = {"model": name, "fine_tuned": tuned, "threads": threads, "runs": b["runs"],
           "warmup": b["warmup"], **percentiles(lat),
           "model_size_mb": model_size_mb(args.model, path, cfg["base_model"]),
           "cpu_memory_mb": rss_mb, "rss_delta_after_load_mb": rss_mb - rss_before / 1e6,
           "hardware": get_hardware_info()}
    log_path = Path(cfg["paths"]["results"]) / "train_log.json"
    if tuned and log_path.exists():
        out["train_seconds"] = json.loads(log_path.read_text())["train_seconds"]
    write_json(f"{cfg['paths']['results']}/bench_{name}.json", out)
    print(format_banner({"model": f"MiniLM ({name})", "fine_tuned": "Yes" if tuned else "No",
                         **out}))


if __name__ == "__main__":
    main()
