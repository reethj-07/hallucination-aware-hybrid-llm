import logging
import os
import pickle
from functools import lru_cache

import faiss
from sentence_transformers import SentenceTransformer

from inference.run_lora_inference import generate_text

logger = logging.getLogger(__name__)

INDEX_PATH = os.getenv("RAG_INDEX_PATH", "rag/faiss_index/index.faiss")
DOCS_PATH = os.getenv("RAG_DOCS_PATH", "rag/faiss_index/docs.pkl")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))


@lru_cache(maxsize=1)
def _load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    return faiss.read_index(INDEX_PATH)


@lru_cache(maxsize=1)
def _load_documents():
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"RAG docs not found at {DOCS_PATH}")
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def run_rag_pipeline(query: str, use_rag: bool = True):
    retrieved_docs = []
    context = ""

    if use_rag:
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

        query_embedding = embedder.encode([query])
        _, I = index.search(query_embedding, k=min(TOP_K, len(documents)))
        retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
        context = "\n".join(retrieved_docs)

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

    answer = generate_text(prompt).strip()

    # 🔒 Hallucination Guard
    if use_rag:
        context_lower = context.lower()
        answer_lower = answer.lower()

        supported = any(
            token in context_lower
            for token in answer_lower.split()
            if len(token) > 5
        )

        if not supported:
            answer = "Not found in retrieved documents"

    return {
        "query": query,
        "answer": answer,
        "used_rag": use_rag,
        "retrieved_documents": retrieved_docs
    }
