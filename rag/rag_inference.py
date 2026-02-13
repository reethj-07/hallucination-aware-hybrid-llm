import hashlib
import logging
import os
import pickle
import re
from collections import OrderedDict
from threading import Lock
from functools import lru_cache

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

from inference.run_lora_inference import generate_text

logger = logging.getLogger(__name__)

INDEX_PATH = os.getenv("RAG_INDEX_PATH", "rag/faiss_index/index.faiss")
DOCS_PATH = os.getenv("RAG_DOCS_PATH", "rag/faiss_index/docs.pkl")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RERANK = os.getenv("RAG_RERANK", "true").lower() == "true"
CANDIDATES_K = int(os.getenv("RAG_CANDIDATES_K", "10"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_DEVICE = os.getenv("RAG_RERANK_DEVICE", "cpu")
RERANK_CACHE_MAX = int(os.getenv("RAG_RERANK_CACHE_MAX", "2048"))


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


@lru_cache(maxsize=1)
def _load_reranker():
    if not RERANK_MODEL:
        raise ValueError("RERANK_MODEL is not configured")
    return CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)


def _extract_text(doc):
    if isinstance(doc, dict):
        return str(doc.get("text", ""))
    return str(doc)


def _hash_doc(doc_text: str) -> str:
    return hashlib.sha1(doc_text.encode("utf-8")).hexdigest()


_rerank_cache: OrderedDict[tuple[str, str], float] = OrderedDict()
_rerank_lock = Lock()


def _cache_get(key: tuple[str, str]) -> float | None:
    with _rerank_lock:
        score = _rerank_cache.get(key)
        if score is not None:
            _rerank_cache.move_to_end(key)
        return score


def _cache_set(key: tuple[str, str], score: float) -> None:
    with _rerank_lock:
        _rerank_cache[key] = score
        _rerank_cache.move_to_end(key)
        if len(_rerank_cache) > RERANK_CACHE_MAX:
            _rerank_cache.popitem(last=False)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def _split_sentences_with_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[.!?]+\s+", text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:
            span_start = text.find(segment, start, end)
            spans.append((span_start, span_start + len(segment), segment))
        start = end
    tail = text[start:].strip()
    if tail:
        span_start = text.find(tail, start)
        spans.append((span_start, span_start + len(tail), tail))
    return spans


def _rerank(query: str, docs: list[str]) -> list[str]:
    if not docs:
        return docs

    try:
        reranker = _load_reranker()
        scores: list[float] = [0.0] * len(docs)
        missing_pairs: list[tuple[str, str]] = []
        missing_indices: list[int] = []

        for idx, doc in enumerate(docs):
            key = (query, _hash_doc(doc))
            cached = _cache_get(key)
            if cached is None:
                missing_pairs.append((query, doc))
                missing_indices.append(idx)
            else:
                scores[idx] = cached

        if missing_pairs:
            predicted = reranker.predict(missing_pairs)
            for score, idx, pair in zip(predicted, missing_indices, missing_pairs):
                scores[idx] = float(score)
                _cache_set((pair[0], _hash_doc(pair[1])), float(score))

        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in ranked]
    except Exception:
        logger.exception("Reranker failed; falling back to original order")
        return docs


def _build_citations(answer: str, docs: list[str]) -> list[dict[str, int | str]]:
    if not answer or not docs:
        return []

    citations: list[dict[str, int | str]] = []
    doc_tokens = [set(_tokenize(doc)) for doc in docs]

    for start, end, sentence in _split_sentences_with_spans(answer):
        sent_tokens = {tok for tok in _tokenize(sentence) if len(tok) > 3}
        if not sent_tokens:
            continue
        best_idx = -1
        best_score = 0
        for idx, tokens in enumerate(doc_tokens):
            score = len(sent_tokens.intersection(tokens))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score > 0:
            citations.append(
                {
                    "start": start,
                    "end": end,
                    "doc_index": best_idx,
                    "snippet": docs[best_idx][:160],
                }
            )

    return citations


def run_rag_pipeline(query: str, use_rag: bool = True):
    retrieved_docs = []
    retrieved_texts: list[str] = []
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

        query_embedding = embedder.encode([query], normalize_embeddings=True)
        candidate_k = min(max(TOP_K, CANDIDATES_K), len(documents))
        _, I = index.search(query_embedding, k=candidate_k)
        retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]

        retrieved_texts = [_extract_text(doc) for doc in retrieved_docs]
        if RERANK:
            retrieved_texts = _rerank(query, retrieved_texts)[:TOP_K]
        else:
            retrieved_texts = retrieved_texts[:TOP_K]

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

    citations = []
    if use_rag and answer != "Not found in retrieved documents":
        citations = _build_citations(answer, retrieved_texts)

    return {
        "query": query,
        "answer": answer,
        "used_rag": use_rag,
        "retrieved_documents": retrieved_texts,
        "citations": citations,
    }
