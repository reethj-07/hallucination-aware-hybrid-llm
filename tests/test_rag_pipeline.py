from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import pytest

from api.schemas import QueryResponse
from rag.hallucination_guard import ABSTENTION_STRING, verify_answer
from rag.rag_inference_lightweight import LightweightPipeline


@pytest.fixture
def pipeline(monkeypatch):
    class DummyEmbedder:
        def encode(self, query: str, normalize_embeddings: bool = True):
            _ = query, normalize_embeddings
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    class DummyIndex:
        def search(self, q, k):
            _ = q
            return np.array([[0.9, 0.8]], dtype=np.float32), np.array([[0, 1]])

    docs = [
        {
            "text": "Rate Limiting: Standard tier allows 1000 requests per minute.",
            "source": "doc_001.txt",
            "chunk_index": 0,
            "total_chunks": 2,
        },
        {
            "text": "Premium tier allows 10000 requests per minute.",
            "source": "doc_001.txt",
            "chunk_index": 1,
            "total_chunks": 2,
        },
    ]

    import rag.shared as shared

    monkeypatch.setattr(shared, "get_embedder", lambda: DummyEmbedder())
    monkeypatch.setattr(shared, "get_faiss_index", lambda: DummyIndex())
    monkeypatch.setattr(shared, "get_docs", lambda: docs)

    return LightweightPipeline()


ADVERSARIAL_QUERIES = [
    ("", pytest.raises(ValueError)),
    ("a" * 5000, pytest.raises(ValueError)),
    ("'; DROP TABLE docs; --", None),
    ("What is 2 + 2?", None),
    ("🤖🚀💡", None),
    ("\x00\x01\x02", None),
]


@pytest.mark.parametrize("query,expectation", ADVERSARIAL_QUERIES)
def test_adversarial_queries(query, expectation, pipeline):
    if expectation is not None:
        with expectation:
            pipeline.embed_query(query)
    else:
        result = asyncio.run(pipeline.arun(query))
        assert result.answer is not None
        assert 0.0 <= result.faithfulness_score <= 1.0


def test_verify_answer_abstains_low_faithfulness(monkeypatch):
    import rag.hallucination_guard as guard

    monkeypatch.setattr(guard, "compute_nli_faithfulness", lambda answer, context: 0.1)
    answer, score, method = verify_answer("confident hallucination", ["unrelated context"], use_nli=True)
    assert answer == ABSTENTION_STRING
    assert score < 0.4
    assert method.endswith("abstained")


def test_chunked_reingestion_increases_chunks(monkeypatch, tmp_path: Path):
    import rag.ingest_docs as ingest

    docs_dir = tmp_path / "rag_docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "a.txt").write_text("sentence " * 1000, encoding="utf-8")
    (docs_dir / "b.txt").write_text("another sentence " * 800, encoding="utf-8")

    index_path = tmp_path / "index.faiss"
    store_path = tmp_path / "docs.pkl"

    class DummyEmbedder:
        def encode(self, texts, normalize_embeddings: bool = True):
            _ = normalize_embeddings
            return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(ingest, "DOCS_PATH", docs_dir)
    monkeypatch.setattr(ingest, "INDEX_PATH", index_path)
    monkeypatch.setattr(ingest, "STORE_PATH", store_path)
    monkeypatch.setattr(ingest, "SentenceTransformer", lambda _: DummyEmbedder())

    ingest.SKIP_CHUNKING = True
    count_skip = ingest.build_index(force_reingest=True)

    ingest.SKIP_CHUNKING = False
    count_chunked = ingest.build_index(force_reingest=True)

    assert count_chunked > count_skip
    assert count_skip == 2


def test_query_response_serialization():
    payload = QueryResponse(
        query="test",
        answer="ok",
        used_rag=True,
        retrieved_documents=["doc"],
        citations=[
            {
                "start": 0,
                "end": 2,
                "doc_index": 0,
                "snippet": "doc",
                "source": "doc_001.txt",
                "chunk_index": 3,
            }
        ],
        faithfulness_score=0.88,
        guard_method="nli_entailment",
        retrieval_scores=[0.77],
        latency_ms=12.3,
        model_mode="lightweight",
    )

    encoded = payload.model_dump()
    assert encoded["faithfulness_score"] == 0.88
    assert encoded["guard_method"] == "nli_entailment"
    assert encoded["retrieval_scores"] == [0.77]
    assert encoded["model_mode"] == "lightweight"
