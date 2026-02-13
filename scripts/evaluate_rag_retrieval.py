"""
Fast evaluation of RAG retrieval quality WITHOUT model loading.
Tests if retrieved documents are relevant to queries.
"""

import json
from pathlib import Path
import re
import os

# Set RAG_RERANK to false to skip model loading
os.environ["RAG_RERANK"] = "false"

from rag.rag_inference import (
    _load_index, _load_documents, _load_embedder, 
    TOP_K, CANDIDATES_K
)

DATASET_PATH = Path("data/eval/qa_pairs.jsonl")


def _load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    records: list[dict] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _normalize_tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok and len(tok) > 2}


def _compute_retrieval_relevance(answer: str, expected_keywords: list[str]) -> tuple[float, float]:
    """
    Compute if retrieved context contains expected keywords.
    - precision: fraction of keywords found in answer
    - recall: fraction of retrieved content containing any keyword
    """
    if not expected_keywords:
        return (1.0, 1.0)

    gold_tokens = _normalize_tokens(" ".join(expected_keywords))
    retrieved_tokens = _normalize_tokens(answer)
    
    if not gold_tokens:
        return (1.0, 1.0)
    if not retrieved_tokens:
        return (0.0, 0.0)

    overlap = gold_tokens.intersection(retrieved_tokens)
    precision = len(overlap) / len(gold_tokens) if gold_tokens else 0.0
    recall = len(overlap) / len(retrieved_tokens) if retrieved_tokens else 0.0
    
    return precision, recall


def evaluate_retrieval():
    print("🔍 Running RAG Retrieval Evaluation...\n")
    
    # Load RAG components (no model loading)
    try:
        index = _load_index()
        documents = _load_documents()
        embedder = _load_embedder()
        print(f"✅ Loaded FAISS index with {len(documents)} documents\n")
    except FileNotFoundError as e:
        print(f"❌ Error loading RAG components: {e}")
        return None
    
    benchmark_queries = _load_dataset()
    
    results = {
        "total": len(benchmark_queries),
        "total_keywords_found": 0,
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "queries": []
    }

    precisions: list[float] = []
    recalls: list[float] = []
    
    for idx, test_case in enumerate(benchmark_queries, 1):
        query = test_case["query"]
        should_find = test_case["should_find"]
        expected_keywords = test_case["expected_keywords"]
        
        # Retrieve documents
        query_embedding = embedder.encode([query], normalize_embeddings=True)
        candidate_k = min(max(TOP_K, CANDIDATES_K), len(documents))
        _, I = index.search(query_embedding, k=candidate_k)
        
        retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
        retrieved_text = "\n".join(
            doc.get("text", "") if isinstance(doc, dict) else str(doc) 
            for doc in retrieved_docs
        )
        
        precision, recall = _compute_retrieval_relevance(retrieved_text, expected_keywords)
        precisions.append(precision)
        recalls.append(recall)
        
        # Check if expected keywords were found
        found_all = all(kw.lower() in retrieved_text.lower() for kw in expected_keywords) if expected_keywords else True
        if found_all and should_find:
            results["total_keywords_found"] += 1
        
        status = "✅" if found_all == should_find else "❌"
        print(f"{status} Q{idx}: {query[:50]}...")
        print(f"   Expected: {'Found' if should_find else 'Not in corpus'}")
        print(f"   Keywords found: {found_all} | Precision: {precision:.2f} | Recall: {recall:.2f}")
        print(f"   Retrieved {len(retrieved_docs)} docs ({len(retrieved_text)} chars)")
        print()
        
        results["queries"].append({
            "query": query,
            "should_find": should_find,
            "found_all_keywords": found_all,
            "precision": precision,
            "recall": recall,
            "retrieved_doc_count": len(retrieved_docs),
            "retrieved_chars": len(retrieved_text),
        })
    
    print("=" * 70)
    print("📊 RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Queries: {results['total']}")
    print(f"Keywords Successfully Retrieved: {results['total_keywords_found']}/{len(benchmark_queries)}")
    success_rate = results['total_keywords_found'] / results['total'] if results['total'] > 0 else 0.0
    print(f"Retrieval Success Rate: {success_rate*100:.1f}%")

    results["avg_precision"] = sum(precisions) / len(precisions) if precisions else 0.0
    results["avg_recall"] = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"Avg Precision (keywords in context): {results['avg_precision']:.2f}")
    print(f"Avg Recall (coverage of keywords): {results['avg_recall']:.2f}")
    
    return results


if __name__ == "__main__":
    results = evaluate_retrieval()
    if results:
        # Save lighter version without full rag pipeline 
        with open("retrieval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to retrieval_results.json")
        print("\n✨ Retrieval evaluation complete. Ready for full RAG pipeline testing.")
