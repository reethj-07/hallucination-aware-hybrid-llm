"""Lightweight RAG pipeline based on retrieval + template extraction."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from rag.base_pipeline import BaseRAGPipeline


class LightweightPipeline(BaseRAGPipeline):
    """Template-based generation. No LLM. ~20-50ms."""

    @property
    def model_mode(self) -> str:
        return "lightweight"

    async def generate(self, query: str, context: str) -> str:
        _ = query
        if not context or len(context.strip()) < 20:
            return "Not found in retrieved documents"

        sentences = [segment.strip() for segment in re.split(r"[.!?]+", context) if segment.strip()]
        if not sentences:
            return context[:200]

        answer = sentences[0]
        if len(sentences) > 1 and len(answer) < 100:
            answer += " " + sentences[1]
        return answer.strip()


def run_rag_pipeline_lightweight(query: str, use_rag: bool = True) -> dict[str, Any]:
    if not use_rag:
        return {
            "query": query,
            "answer": "RAG disabled",
            "used_rag": False,
            "retrieved_documents": [],
            "citations": [],
            "faithfulness_score": 1.0,
            "guard_method": "disabled",
            "retrieval_scores": [],
            "model_mode": "lightweight",
        }

    pipeline = LightweightPipeline()
    result = asyncio.run(pipeline.arun(query))
    return {
        "query": result.query,
        "answer": result.answer,
        "used_rag": result.used_rag,
        "retrieved_documents": result.retrieved_documents,
        "citations": result.citations,
        "faithfulness_score": result.faithfulness_score,
        "guard_method": result.guard_method,
        "retrieval_scores": result.retrieval_scores,
        "latency_ms": result.latency_ms,
        "model_mode": result.model_mode,
    }
