from fastapi.testclient import TestClient

import api.main as api_main


def test_health():
    client = TestClient(api_main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate(monkeypatch):
    client = TestClient(api_main.app)
    monkeypatch.setattr(api_main, "generate_text", lambda prompt, max_new_tokens=120: "ok")
    response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 5})
    assert response.status_code == 200
    assert response.json() == {"output": "ok"}


def test_query(monkeypatch):
    client = TestClient(api_main.app)
    monkeypatch.setattr(
        api_main,
        "run_rag_pipeline",
        lambda query, use_rag=True: {
            "query": query,
            "answer": "ok",
            "used_rag": use_rag,
            "retrieved_documents": [],
        },
    )
    response = client.post("/query", json={"query": "hi", "use_rag": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "ok"
    assert payload["query"] == "hi"
    assert payload["used_rag"] is True
    assert payload["retrieved_documents"] == []
    assert payload["citations"] == []
