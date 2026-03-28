"""Shared singleton loaders for FAISS index, embedder, and reranker."""

from __future__ import annotations

import pickle
from functools import lru_cache

import faiss
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    from api.config import get_settings

    return SentenceTransformer(get_settings().rag_embed_model)


@lru_cache(maxsize=1)
def get_faiss_index() -> faiss.Index:
    from api.config import get_settings

    path = get_settings().rag_index_path
    return faiss.read_index(str(path))


@lru_cache(maxsize=1)
def get_docs() -> list[dict]:
    from api.config import get_settings

    path = get_settings().rag_docs_path
    with path.open("rb") as handle:
        return pickle.load(handle)
