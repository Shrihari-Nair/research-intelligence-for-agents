"""FastAPI app: POST /runs, GET /runs, GET /runs/{id}, SSE events, report, health."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from research_os.api.routes_runs import router as runs_router
from research_os.api.service import RunService
from research_os.common import load_config, load_dotenv

log = logging.getLogger(__name__)


def build_default_service(cfg: dict | None = None) -> RunService:
    """Real components from config: SetFit router, supervisor, agents, tool registry, Gemini or the offline demo LLM."""
    from research_os.agents import build_agents
    from research_os.agents.supervisor import Supervisor
    from research_os.llm.demo_client import DemoLLM
    from research_os.llm.gemini_client import GeminiLLMClient
    from research_os.llm.models import model_for_role, provider_settings
    from research_os.router.inference import ResearchRouter
    from research_os.tools import build_registry

    load_dotenv()
    cfg = cfg or load_config("configs/api.yaml")
    prov = provider_settings()
    want = os.environ.get("RESEARCH_OS_LLM", cfg["llm"])
    use_gemini = want == "gemini" or (want == "auto" and bool(os.environ.get(prov["api_key_env"])))
    if use_gemini:
        llm_factory = lambda: GeminiLLMClient(model_for_role("supervisor"), prov["api_key_env"], prov["timeout_s"])  # noqa: E731
        backend = "gemini"
    else:
        llm_factory, backend = DemoLLM, DemoLLM.BACKEND
        log.warning("running with the offline DEMO LLM (set GOOGLE_API_KEY and RESEARCH_OS_LLM=gemini for real answers)")
    from research_os.harness.approvals import ApprovalPolicy
    from research_os.memory import memory_cfg
    from research_os.memory.research_store import ResearchStore

    mcfg = memory_cfg()
    url = os.environ.get(mcfg["database_url_env"]) or f"sqlite:///{mcfg['sqlite_path']}"
    store = ResearchStore(url)
    store.migrate()
    return RunService(router=ResearchRouter.from_config(os.environ.get("RESEARCH_OS_ROUTER_MODEL")), supervisor=Supervisor(),
                      agents=build_agents(), llm_factory=llm_factory, tools_factory=lambda: build_registry(memory=store),
                      runs_dir=cfg["runs_dir"], llm_backend=backend, max_workers=cfg["max_workers"], memory=store,
                      approvals=ApprovalPolicy.from_config() if cfg.get("approvals", True) else None)


def create_app(service: RunService | None = None, cors_origins: list[str] | None = None, max_text_chars: int | None = None) -> FastAPI:
    """Build the app. Pass `service` to inject a prebuilt RunService (tests); otherwise it is built at startup."""
    cfg = load_config("configs/api.yaml")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or build_default_service(cfg)
        yield
        app.state.service.shutdown()

    app = FastAPI(title="Research Intelligence OS", version="0.1.0", lifespan=lifespan)
    app.state.max_text_chars = max_text_chars or cfg["max_text_chars"]
    env_origins = [o.strip() for o in os.environ.get("RESEARCH_OS_CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins if cors_origins is not None else (env_origins or cfg["cors_origins"]),
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(runs_router)

    @app.get("/health")
    def health() -> dict:
        s = getattr(app.state, "service", None)
        return {"status": "ok", "llm_backend": s.llm_backend if s else None, "runs": len(s.list()) if s else 0}

    return app


def __getattr__(name: str):
    """`uvicorn research_os.api.app:app` builds the real app lazily."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
