"""Wiring helpers shared by demo, API and comparison scripts."""
from __future__ import annotations

import logging
import os

from research_os.common import load_config, load_dotenv, load_intents
from research_os.legacy.agents.tools import ToolBox
from research_os.legacy.graph.workflow import build_graph
from research_os.llm.gemini_client import GeminiLLMClient
from research_os.llm.mock_client import MockLLMClient
from research_os.llm.models import model_for_role, provider_settings
from research_os.router.loader import load_router
from research_os.router.systems.hybrid import HybridSystem
from research_os.router.systems.llm_only import LLMOnlySystem
from research_os.router.systems.minilm_only import MiniLMOnlySystem

log = logging.getLogger(__name__)
HEALTHCARE_INTENTS = "configs/healthcare/intents.yaml"


def build_llm(rcfg: dict):
    """Return (client, backend_name): Gemini if the key env var is set, else the mock."""
    load_dotenv()
    prov = provider_settings()
    if os.environ.get(prov["api_key_env"]):
        return GeminiLLMClient(model_for_role("router_fallback"), prov["api_key_env"], prov["timeout_s"]), "gemini"
    log.warning("%s not set: using the offline mock LLM", prov["api_key_env"])
    return MockLLMClient(), "mock"


def build_system(kind: str, router, llm, rcfg: dict):
    """kind in {'hybrid','minilm','llm'}."""
    if kind == "hybrid":
        return HybridSystem(router, llm, rcfg, load_intents(HEALTHCARE_INTENTS))
    if kind == "minilm":
        return MiniLMOnlySystem(router, rcfg)
    if kind == "llm":
        return LLMOnlySystem(llm, rcfg, load_intents(HEALTHCARE_INTENTS))
    raise ValueError(f"unknown system kind {kind}")


def build_app_graph(model_path: str = "models/setfit_router", kind: str = "hybrid",
                    routing_cfg: str = "configs/routing.yaml"):
    """Load model + LLM, return (graph, system, llm_backend_name)."""
    rcfg = load_config(routing_cfg)
    router, _, _ = load_router(model_path)
    llm, backend = build_llm(rcfg)
    system = build_system(kind, router, llm, rcfg)
    return build_graph(system, ToolBox(), llm), system, backend
