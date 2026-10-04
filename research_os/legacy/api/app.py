"""FastAPI service: POST /chat, GET /health. Model is loaded once at startup."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, StringConstraints

from research_os.common import load_config
from research_os.legacy.graph.workflow import run_text

log = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    """Routing + agent result for one utterance."""

    intent: str
    confidence: float
    source: str
    agent: str
    response: str
    latency_ms: float


def create_app(graph=None, llm_backend: Optional[str] = None, max_chars: Optional[int] = None) -> FastAPI:
    """Build the app. Pass `graph` to inject a prebuilt workflow (tests); else load at startup."""
    if max_chars is None:
        max_chars = load_config("configs/routing.yaml")["max_text_chars"]
    Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=max_chars)]

    class ChatRequest(BaseModel):
        text: Text = Field(description="User utterance")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if graph is not None:
            app.state.graph, app.state.backend = graph, llm_backend or "injected"
        else:
            from research_os.legacy.factory import build_app_graph
            app.state.graph, _, app.state.backend = build_app_graph()
        log.info("model loaded; llm backend=%s", app.state.backend)
        yield

    app = FastAPI(title="Agentic intent router", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "model_loaded": request.app.state.graph is not None,
                "llm_backend": request.app.state.backend}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, request: Request) -> ChatResponse:
        state = run_text(request.app.state.graph, req.text)
        r = state["routing"]
        return ChatResponse(intent=r["intent"], confidence=r["confidence"], source=r["source"],
                            agent=state["agent_name"], response=state["response"],
                            latency_ms=round(r["latency_ms"], 2))

    return app


def __getattr__(name: str):
    """`uvicorn research_os.legacy.api.app:app` builds the real app lazily."""
    if name == "app":
        return create_app()
    raise AttributeError(name)
