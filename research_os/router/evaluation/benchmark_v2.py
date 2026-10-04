"""Latency of the research router v2: classifier alone vs full route() (classifier + entity extraction)."""
from __future__ import annotations

import argparse
import logging
import time

import torch

from research_os.common import get_hardware_info, load_config, setup_logging, write_json
from research_os.router.data.load import load_split
from research_os.router.evaluation.benchmark import percentiles
from research_os.router.inference import ResearchRouter

log = logging.getLogger(__name__)


def _time(fn, texts: list[str], warmup: int, runs: int) -> dict:
    for i in range(warmup):
        fn(texts[i % len(texts)])
    lat = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(texts[i % len(texts)])
        lat.append((time.perf_counter() - t0) * 1000)
    return percentiles(lat)


def benchmark_router_v2(router: ResearchRouter, texts: list[str], warmup: int, runs: int) -> dict:
    """Percentiles for classifier-only, entities-only and the full route(); overhead = full - predict_only."""
    predict = _time(router.intent_router.predict, texts, warmup, runs)
    ents = _time(router.entities.extract, texts, warmup, runs)
    full = _time(router.route, texts, warmup, runs)
    return {"runs": runs, "warmup": warmup, "predict_only": predict, "entities_only": ents, "full_route": full,
            "p50_overhead_ms": full["p50_ms"] - predict["p50_ms"], "p95_overhead_ms": full["p95_ms"] - predict["p95_ms"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training.yaml")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    b = cfg["benchmark"]
    torch.set_num_threads(b["threads"])
    router = ResearchRouter.from_config(training_cfg=args.config)
    texts = load_split("test", cfg["paths"]["processed"])["text"].tolist()
    res = benchmark_router_v2(router, texts, b["warmup"], b["runs"])
    res.update({"threads": b["threads"], "hardware": get_hardware_info(), "threshold": router.threshold})
    write_json(f"{cfg['paths']['results']}/bench_router_v2.json", res)
    log.info("predict-only P95 %.2f ms | full route P95 %.2f ms | entities-only P95 %.3f ms | P95 overhead %.3f ms",
             res["predict_only"]["p95_ms"], res["full_route"]["p95_ms"], res["entities_only"]["p95_ms"], res["p95_overhead_ms"])


if __name__ == "__main__":
    main()
