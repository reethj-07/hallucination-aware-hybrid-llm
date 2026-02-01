import numpy as np
import pytest

import rag.rag_inference as rag_inference


def test_run_rag_pipeline_success(monkeypatch):
    class DummyIndex:
        def search(self, q_emb, k=3):
            return None, np.array([[0]])

    def fake_embedder():
        class DummyEmbedder:
            def encode(self, texts):
                return np.zeros((1, 3), dtype=np.float32)
        return DummyEmbedder()

    doc_text = "This document contains information about machine learning"
    monkeypatch.setattr(rag_inference, "_load_index", lambda: DummyIndex())
    monkeypatch.setattr(rag_inference, "_load_documents", lambda: [doc_text])
    monkeypatch.setattr(rag_inference, "_load_embedder", fake_embedder)
    monkeypatch.setattr(rag_inference, "generate_text", lambda prompt: "Machine learning is in this document information")

    result = rag_inference.run_rag_pipeline("query", use_rag=True)
    assert result["used_rag"] is True
    assert "information" in result["answer"].lower() or "machine" in result["answer"].lower()
    assert result["retrieved_documents"] == [doc_text]


def test_run_rag_pipeline_missing_resources(monkeypatch):
    def raise_missing():
        raise FileNotFoundError("missing")

    monkeypatch.setattr(rag_inference, "_load_index", raise_missing)

    result = rag_inference.run_rag_pipeline("query", use_rag=True)
    assert result["used_rag"] is False
    assert result["answer"] == "Not found in retrieved documents"
    assert "error" in result
