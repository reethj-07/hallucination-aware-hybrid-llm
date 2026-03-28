from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rag.hallucination_guard import ABSTENTION_STRING, verify_answer


@dataclass
class RAGResult:
    query: str
    answer: str
    used_rag: bool
    retrieved_documents: list[str]
    citations: list[dict]
    faithfulness_score: float = 0.0
    guard_method: str = "none"
    retrieval_scores: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    model_mode: str = "lightweight"


class BaseRAGPipeline(ABC):
    """Shared retrieval and verification behavior for all RAG pipelines."""

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if len(query) > 4096:
            raise ValueError("Query exceeds max length")

        from rag.shared import get_embedder

        return get_embedder().encode(query, normalize_embeddings=True).tolist()

    def retrieve(self, query_embedding: list[float], top_k: int) -> tuple[list[dict], list[float]]:
        from rag.shared import get_docs, get_faiss_index

        index = get_faiss_index()
        docs_meta = get_docs()
        if not docs_meta:
            return [], []

        k = min(max(top_k, 1), len(docs_meta))
        query_arr = np.array([query_embedding], dtype=np.float32)
        scores, indices = index.search(query_arr, k)

        retrieved: list[dict] = [
            docs_meta[i] for i in indices[0] if isinstance(i, (int, np.integer)) and i < len(docs_meta)
        ]
        score_list = [float(score) for score in scores[0].tolist()]
        return retrieved, score_list

    @abstractmethod
    async def generate(self, query: str, context: str) -> str:
        ...

    @property
    @abstractmethod
    def model_mode(self) -> str:
        ...

    def _build_context(self, docs: list[str], max_chars: int) -> str:
        context_parts: list[str] = []
        used = 0
        for doc in docs:
            if not doc:
                continue
            if used + len(doc) + 1 > max_chars:
                break
            context_parts.append(doc)
            used += len(doc) + 1
        return "\n".join(context_parts)

    def _extract_citations(self, answer: str, docs_meta: list[dict]) -> list[dict]:
        citations: list[dict] = []
        answer_words = {w.lower() for w in answer.split()[:15] if w}
        if not answer_words:
            return citations

        for idx, doc in enumerate(docs_meta):
            text = str(doc.get("text", ""))
            doc_words = {w.lower() for w in text.split()[:80] if w}
            if answer_words.intersection(doc_words):
                citations.append(
                    {
                        "start": 0,
                        "end": min(len(answer), 120),
                        "doc_index": idx,
                        "snippet": text[:200],
                        "source": doc.get("source"),
                        "chunk_index": doc.get("chunk_index"),
                    }
                )
        return citations

    async def arun(self, query: str, top_k: int = 5) -> RAGResult:
        from api.config import get_settings

        settings = get_settings()

        query_embedding = self.embed_query(query)
        docs_meta, scores = self.retrieve(query_embedding, top_k)
        docs = [str(doc.get("text", "")) for doc in docs_meta]

        if not docs:
            return RAGResult(
                query=query,
                answer=ABSTENTION_STRING,
                used_rag=True,
                retrieved_documents=[],
                citations=[],
                faithfulness_score=1.0,
                guard_method="no_docs_retrieved",
                retrieval_scores=[],
                model_mode=self.model_mode,
            )

        context = self._build_context(docs, settings.rag_max_context_chars)
        raw_answer = await self.generate(query, context)
        final_answer, faith_score, method = verify_answer(
            raw_answer,
            docs,
            use_nli=settings.use_nli_guard,
        )

        return RAGResult(
            query=query,
            answer=final_answer,
            used_rag=True,
            retrieved_documents=docs,
            citations=self._extract_citations(final_answer, docs_meta),
            faithfulness_score=faith_score,
            guard_method=method,
            retrieval_scores=scores,
            model_mode=self.model_mode,
        )
