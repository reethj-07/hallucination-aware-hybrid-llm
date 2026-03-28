from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from api.config import Settings, get_settings
from api.dependencies import get_rag_pipeline, verify_api_key
from api.metrics import (
    rag_abstentions_total,
    rag_faithfulness_score,
    rag_retrieval_success_total,
)
from api.schemas import QueryRequest, QueryResponse
from rag.base_pipeline import RAGResult
from rag.hallucination_guard import ABSTENTION_STRING

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    pipeline=Depends(get_rag_pipeline),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    if not request.use_rag:
        return QueryResponse(
            query=request.query,
            answer="RAG disabled",
            used_rag=False,
            retrieved_documents=[],
            citations=[],
            faithfulness_score=1.0,
            guard_method="disabled",
            retrieval_scores=[],
            latency_ms=0.0,
            model_mode="lightweight" if settings.rag_lightweight else "full",
        )

    start = time.perf_counter()
    result: RAGResult = await pipeline.arun(request.query, top_k=request.top_k or settings.rag_top_k)
    result.latency_ms = round((time.perf_counter() - start) * 1000, 2)

    rag_faithfulness_score.observe(result.faithfulness_score)
    if result.answer == ABSTENTION_STRING:
        rag_abstentions_total.labels(guard_method=result.guard_method).inc()
    if result.retrieved_documents:
        rag_retrieval_success_total.labels(model_mode=result.model_mode).inc()

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        used_rag=result.used_rag,
        retrieved_documents=result.retrieved_documents,
        citations=result.citations,
        faithfulness_score=result.faithfulness_score,
        guard_method=result.guard_method,
        retrieval_scores=result.retrieval_scores,
        latency_ms=result.latency_ms,
        model_mode=result.model_mode,
    )
