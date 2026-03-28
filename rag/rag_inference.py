from __future__ import annotations

import asyncio
from typing import Any

from inference.run_lora_inference import generate_text
from rag.base_pipeline import BaseRAGPipeline


class FullPipeline(BaseRAGPipeline):
    """Phi-3 + LoRA generation. ~150-700ms."""

    @property
    def model_mode(self) -> str:
        return "full"

    async def generate(self, query: str, context: str) -> str:
        prompt = f"""
You are a senior machine learning engineer answering a technical question.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT use prior knowledge
- Do NOT repeat the question
- If the answer is not present in the context, reply EXACTLY:
  "Not found in retrieved documents"

Context:
{context}

Question:
{query}

Final Answer:
"""
        return generate_text(prompt).strip()


def run_rag_pipeline(query: str, use_rag: bool = True) -> dict[str, Any]:
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
            "model_mode": "full",
        }

    pipeline = FullPipeline()
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
