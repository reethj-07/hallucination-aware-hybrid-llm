from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User query")
    use_rag: bool = Field(default=True)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class CitationSpan(BaseModel):
    start: int
    end: int
    doc_index: int
    snippet: str
    source: Optional[str] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    used_rag: bool
    retrieved_documents: list[str]
    citations: list[CitationSpan]
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    guard_method: str = Field(default="none")
    retrieval_scores: list[float] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0)
    model_mode: str = Field(default="lightweight")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(120, ge=1, le=512)


class GenerateResponse(BaseModel):
    output: str


class HealthResponse(BaseModel):
    status: str
    faiss_index_loaded: bool
    embedder_loaded: bool
    lightweight_mode: bool
