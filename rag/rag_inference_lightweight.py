"""
Lightweight RAG pipeline for testing - uses retrieval + template-based answer generation
instead of heavy LLM model loading.
"""

import os
import pickle
import logging
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

INDEX_PATH = os.getenv("RAG_INDEX_PATH", "rag/faiss_index/index.faiss")
DOCS_PATH = os.getenv("RAG_DOCS_PATH", "rag/faiss_index/docs.pkl")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))

_index = None
_documents = None
_embedder = None


def _load_index():
    global _index
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        _index = faiss.read_index(INDEX_PATH)
    return _index


def _load_documents():
    global _documents
    if _documents is None:
        if not os.path.exists(DOCS_PATH):
            raise FileNotFoundError(f"RAG docs not found at {DOCS_PATH}")
        with open(DOCS_PATH, "rb") as f:
            _documents = pickle.load(f)
    return _documents


def _load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _extract_text(doc):
    if isinstance(doc, dict):
        return doc.get("text", "")
    return str(doc)


def run_rag_pipeline_lightweight(query: str, use_rag: bool = True) -> dict:
    """
    Lightweight RAG: fast retrieval + template-based response generation.
    Skips LLM model loading for quick testing.
    """
    retrieved_docs = []
    retrieved_texts: list[str] = []

    if not use_rag:
        return {
            "query": query,
            "answer": "RAG disabled",
            "used_rag": False,
            "retrieved_documents": [],
        }

    try:
        index = _load_index()
        documents = _load_documents()
        embedder = _load_embedder()
    except FileNotFoundError as exc:
        logger.error("RAG resources missing: %s", exc)
        return {
            "query": query,
            "answer": "Not found in retrieved documents",
            "used_rag": False,
            "retrieved_documents": [],
            "error": str(exc),
        }

    if not documents:
        logger.warning("RAG documents list is empty")
        return {
            "query": query,
            "answer": "Not found in retrieved documents",
            "used_rag": True,
            "retrieved_documents": [],
            "error": "RAG documents are empty",
        }

    # Retrieve candidate documents
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    _, I = index.search(query_embedding, k=min(TOP_K * 3, len(documents)))
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    retrieved_texts = [_extract_text(doc) for doc in retrieved_docs][:TOP_K]

    # Build context (budget: MAX_CONTEXT_CHARS)
    context_parts = []
    total_chars = 0
    for doc_text in retrieved_texts:
        if not doc_text:
            continue
        if total_chars + len(doc_text) + 1 > MAX_CONTEXT_CHARS:
            break
        context_parts.append(doc_text)
        total_chars += len(doc_text) + 1
    context = "\n".join(context_parts)

    # Template-based answer generation
    if not context or len(context.strip()) < 20:
        answer = "Not found in retrieved documents"
    else:
        # Extract key sentences from context
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]
        if sentences:
            # Use first 1-2 relevant sentences
            answer = sentences[0]
            if len(sentences) > 1 and len(answer) < 100:
                answer += " " + sentences[1]
            answer = answer.strip()
        else:
            answer = context[:200]

    # Hallucination guard: verify answer uses context
    context_lower = context.lower()
    answer_lower = answer.lower()
    if not any(token in context_lower for token in answer_lower.split() if len(token) > 5):
        answer = "Not found in retrieved documents"

    return {
        "query": query,
        "answer": answer,
        "used_rag": True,
        "retrieved_documents": retrieved_texts,
        "citations": [],
    }


if __name__ == "__main__":
    # Quick test
    query = "What is the rate limit for Standard tier API?"
    result = run_rag_pipeline_lightweight(query)
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Docs retrieved: {len(result['retrieved_documents'])}")
