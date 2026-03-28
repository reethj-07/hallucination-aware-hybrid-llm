from __future__ import annotations

from fastapi import APIRouter

from api.config import get_settings
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    from rag import shared

    settings = get_settings()
    faiss_ok = True
    embedder_ok = True

    try:
        shared.get_faiss_index()
    except Exception:
        faiss_ok = False

    try:
        shared.get_embedder()
    except Exception:
        embedder_ok = False

    return HealthResponse(
        status="ok",
        faiss_index_loaded=faiss_ok,
        embedder_loaded=embedder_ok,
        lightweight_mode=settings.rag_lightweight,
    )
