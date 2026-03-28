from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main
from api.dependencies import get_rag_pipeline
from rag.base_pipeline import RAGResult


class DummyPipeline:
    async def arun(self, query: str, top_k: int = 5) -> RAGResult:
        _ = top_k
        return RAGResult(
            query=query,
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
                    "chunk_index": 0,
                }
            ],
            faithfulness_score=0.9,
            guard_method="nli_entailment",
            retrieval_scores=[0.7],
            model_mode="lightweight",
        )


client = TestClient(api_main.app)


def setup_function():
    api_main.app.dependency_overrides[get_rag_pipeline] = lambda: DummyPipeline()


def teardown_function():
    api_main.app.dependency_overrides.clear()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "faiss_index_loaded" in payload
    assert "embedder_loaded" in payload
    assert "lightweight_mode" in payload


def test_query_schema_fields():
    response = client.post("/query", json={"query": "hi", "use_rag": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "ok"
    assert payload["query"] == "hi"
    assert payload["used_rag"] is True
    assert payload["retrieved_documents"] == ["doc"]
    assert payload["faithfulness_score"] == 0.9
    assert payload["guard_method"] == "nli_entailment"
    assert payload["retrieval_scores"] == [0.7]
    assert payload["model_mode"] == "lightweight"


def test_query_rejects_empty_query():
    response = client.post("/query", json={"query": "", "use_rag": True})
    assert response.status_code == 422
