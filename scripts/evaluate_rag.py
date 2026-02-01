"""
Evaluation benchmark for production RAG system.
Tests retrieval quality and hallucination prevention.
"""

import json
from rag.rag_inference import run_rag_pipeline

BENCHMARK_QUERIES = [
    {
        "query": "What is the rate limit for Standard tier API?",
        "expected_keywords": ["1000", "requests", "minute"],
        "should_find": True
    },
    {
        "query": "How do I connect to a PostgreSQL database?",
        "expected_keywords": ["PostgreSQL", "connection", "SSL/TLS"],
        "should_find": True
    },
    {
        "query": "What error code is returned when rate limit exceeded?",
        "expected_keywords": ["429", "Too Many Requests"],
        "should_find": True
    },
    {
        "query": "What is the RTO in disaster recovery?",
        "expected_keywords": ["15 minutes", "Recovery Time"],
        "should_find": True
    },
    {
        "query": "How do I enable MFA on my account?",
        "expected_keywords": ["MFA", "security"],
        "should_find": True
    },
    {
        "query": "What quantum computing features are available?",
        "expected_keywords": [],
        "should_find": False  # Should abstain - not in docs
    },
    {
        "query": "What is SOC 2 certification about?",
        "expected_keywords": ["SOC 2", "compliance"],
        "should_find": True
    },
    {
        "query": "How do I export data to S3?",
        "expected_keywords": ["export", "S3"],
        "should_find": True
    },
    {
        "query": "What is the data retention for Premium tier?",
        "expected_keywords": ["90 days", "retention"],
        "should_find": True
    },
    {
        "query": "How do I configure webhooks?",
        "expected_keywords": ["webhook", "Settings"],
        "should_find": True
    },
]

def run_benchmark():
    print("🧪 Running RAG Evaluation Benchmark...\n")
    
    results = {
        "total": len(BENCHMARK_QUERIES),
        "correct_retrieval": 0,
        "correct_abstention": 0,
        "hallucinations_prevented": 0,
        "queries": []
    }
    
    for idx, test_case in enumerate(BENCHMARK_QUERIES, 1):
        query = test_case["query"]
        should_find = test_case["should_find"]
        expected_keywords = test_case["expected_keywords"]
        
        result = run_rag_pipeline(query, use_rag=True)
        answer = result["answer"].lower()
        found_answer = "not found in retrieved documents" not in answer
        
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
        
        status = "✅" if success else "❌"
        print(f"{status} Q{idx}: {query[:60]}...")
        print(f"   Expected: {'Found' if should_find else 'Abstain'} | Got: {'Found' if found_answer else 'Abstained'}")
        if should_find:
            print(f"   Keywords: {', '.join(expected_keywords)}")
        print()
        
        results["queries"].append({
            "query": query,
            "answer_snippet": answer[:100],
            "success": success,
            "should_find": should_find,
            "found": found_answer
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
    
    return results

if __name__ == "__main__":
    results = run_benchmark()
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to benchmark_results.json")
