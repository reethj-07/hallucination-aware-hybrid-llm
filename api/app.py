from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.metrics import setup_metrics
from api.middleware import setup_middleware
from api.routers.health import router as health_router
from api.routers.query import router as query_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Hallucination-Aware RAG API",
        version="2.0.0",
        description="Production-grade RAG with NLI faithfulness verification",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    setup_middleware(app, settings)
    setup_metrics(app)
    app.include_router(query_router)
    app.include_router(health_router)
    return app


app = create_app()
