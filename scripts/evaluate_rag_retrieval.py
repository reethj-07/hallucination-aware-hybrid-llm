"""Fast retrieval-only evaluation for the RAG index."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from api.config import get_settings
from rag.shared import get_docs, get_embedder, get_faiss_index

DATASET_PATH = Path("data/eval/qa_pairs.jsonl")


def _load_dataset() -> list[dict]:
    return [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token and len(token) > 2}


def _compute_retrieval_relevance(retrieved_text: str, expected_keywords: list[str]) -> tuple[float, float]:
    if not expected_keywords:
        return 1.0, 1.0

    gold_tokens = _normalize_tokens(" ".join(expected_keywords))
    retrieved_tokens = _normalize_tokens(retrieved_text)
    if not gold_tokens:
        return 1.0, 1.0
    if not retrieved_tokens:
        return 0.0, 0.0

    overlap = gold_tokens.intersection(retrieved_tokens)
    precision = len(overlap) / len(gold_tokens)
    recall = len(overlap) / len(retrieved_tokens)
    return precision, recall


def evaluate_retrieval() -> dict:
    settings = get_settings()
    index = get_faiss_index()
    documents = get_docs()
    embedder = get_embedder()

    benchmark_queries = _load_dataset()
    top_k = settings.rag_top_k

    results = {
        "total": len(benchmark_queries),
        "total_keywords_found": 0,
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "queries": [],
    }

    precisions: list[float] = []
    recalls: list[float] = []

    for case in benchmark_queries:
        query = case.get("query", case.get("question", ""))
        expected_keywords = case.get("expected_keywords", [])
        should_find = bool(case.get("should_find", True))

        query_embedding = embedder.encode([query], normalize_embeddings=True)
        _, indices = index.search(np.asarray(query_embedding, dtype=np.float32), k=min(top_k, len(documents)))

        retrieved_docs = [documents[i] for i in indices[0] if i < len(documents)]
        retrieved_text = "\n".join(str(doc.get("text", "")) for doc in retrieved_docs)

        precision, recall = _compute_retrieval_relevance(retrieved_text, expected_keywords)
        precisions.append(precision)
        recalls.append(recall)

        found_all = all(keyword.lower() in retrieved_text.lower() for keyword in expected_keywords) if expected_keywords else True
        if found_all and should_find:
            results["total_keywords_found"] += 1

        results["queries"].append(
            {
                "query": query,
                "should_find": should_find,
                "found_all_keywords": found_all,
                "precision": precision,
                "recall": recall,
                "retrieved_doc_count": len(retrieved_docs),
                "retrieved_chars": len(retrieved_text),
            }
        )

    results["avg_precision"] = sum(precisions) / len(precisions) if precisions else 0.0
    results["avg_recall"] = sum(recalls) / len(recalls) if recalls else 0.0
    return results


if __name__ == "__main__":
    output = evaluate_retrieval()
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results/retrieval_results_v2.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")
