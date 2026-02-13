"""
Evaluation benchmark for production RAG system.
Tests retrieval quality and hallucination prevention.
"""

import json
from pathlib import Path

import re

from rag.rag_inference import run_rag_pipeline

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


def _normalize_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def _compute_em_f1(answer: str, expected_keywords: list[str], abstained: bool) -> tuple[float, float]:
    if not expected_keywords:
        return (1.0, 1.0) if abstained else (0.0, 0.0)

    gold_tokens = _normalize_tokens(" ".join(expected_keywords))
    pred_tokens = _normalize_tokens(answer)
    if not gold_tokens or not pred_tokens:
        return 0.0, 0.0

    gold_set = set(gold_tokens)
    pred_set = set(pred_tokens)
    overlap = len(gold_set.intersection(pred_set))
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    em = 1.0 if overlap == len(gold_set) else 0.0
    return em, f1


def _citation_precision(answer: str, citations: list[dict], docs: list[str]) -> float:
    if not citations:
        return 0.0

    answer_tokens = _normalize_tokens(answer)
    if not answer_tokens:
        return 0.0

    correct = 0
    for citation in citations:
        doc_index = citation.get("doc_index")
        if doc_index is None or doc_index >= len(docs):
            continue
        doc_tokens = set(_normalize_tokens(docs[doc_index]))
        span_text = answer[citation.get("start", 0):citation.get("end", 0)]
        span_tokens = {tok for tok in _normalize_tokens(span_text) if len(tok) > 3}
        if span_tokens and span_tokens.intersection(doc_tokens):
            correct += 1
    return correct / len(citations)

def run_benchmark():
    print("🧪 Running RAG Evaluation Benchmark...\n")

    benchmark_queries = _load_dataset()
    
    results = {
        "total": len(benchmark_queries),
        "correct_retrieval": 0,
        "correct_abstention": 0,
        "hallucinations_prevented": 0,
        "avg_em": 0.0,
        "avg_f1": 0.0,
        "avg_citation_precision": 0.0,
        "queries": []
    }

    em_scores: list[float] = []
    f1_scores: list[float] = []
    citation_scores: list[float] = []
    
    for idx, test_case in enumerate(benchmark_queries, 1):
        query = test_case["query"]
        should_find = test_case["should_find"]
        expected_keywords = test_case["expected_keywords"]
        
        result = run_rag_pipeline(query, use_rag=True)
        answer_text = result["answer"]
        answer = answer_text.lower()
        found_answer = "not found in retrieved documents" not in answer
        abstained = not found_answer
        
        # Check if retrieval decision is correct
        correct_decision = found_answer == should_find
        
        # Check if answer contains expected keywords
        contains_keywords = all(kw.lower() in answer for kw in expected_keywords) if expected_keywords else True
        
        success = correct_decision and (contains_keywords if should_find else True)
        
        if should_find:
            if success:
                results["correct_retrieval"] += 1
            else:
                results["hallucinations_prevented"] += 1 if not found_answer else 0
        else:
            if success:
                results["correct_abstention"] += 1
        
        em, f1 = _compute_em_f1(answer, expected_keywords, abstained)
        citation_precision = _citation_precision(
            answer_text,
            result.get("citations", []),
            result.get("retrieved_documents", []),
        )
        em_scores.append(em)
        f1_scores.append(f1)
        citation_scores.append(citation_precision)

        status = "✅" if success else "❌"
        print(f"{status} Q{idx}: {query[:60]}...")
        print(f"   Expected: {'Found' if should_find else 'Abstain'} | Got: {'Found' if found_answer else 'Abstained'}")
        if should_find:
            print(f"   Keywords: {', '.join(expected_keywords)}")
        print(f"   EM: {em:.2f} | F1: {f1:.2f} | Citation precision: {citation_precision:.2f}")
        print()
        
        results["queries"].append({
            "query": query,
            "answer_snippet": answer_text[:100],
            "success": success,
            "should_find": should_find,
            "found": found_answer,
            "em": em,
            "f1": f1,
            "citation_precision": citation_precision,
        })
    
    print("=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Queries: {results['total']}")
    print(f"Correct Retrievals: {results['correct_retrieval']}")
    print(f"Correct Abstentions: {results['correct_abstention']}")
    print(f"Hallucinations Prevented: {results['hallucinations_prevented']}")
    accuracy = (results['correct_retrieval'] + results['correct_abstention']) / results['total']
    print(f"Accuracy: {accuracy*100:.1f}%")

    results["avg_em"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
    results["avg_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    results["avg_citation_precision"] = (
        sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
    )
    print(f"Avg EM: {results['avg_em']:.2f}")
    print(f"Avg F1: {results['avg_f1']:.2f}")
    print(f"Avg Citation Precision: {results['avg_citation_precision']:.2f}")
    
    return results

if __name__ == "__main__":
    results = run_benchmark()
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to benchmark_results.json")
