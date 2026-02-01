"""
Production RAG Evaluation (lightweight - no model loading).
Tests retrieval quality directly without inference.
"""

import json
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

BENCHMARK_QUERIES = [
    {
        "query": "What is the rate limit for Standard tier API?",
        "expected_keywords": ["1000", "requests", "minute"],
        "should_find": True
    },
    {
        "query": "How do I connect to a PostgreSQL database?",
        "expected_keywords": ["PostgreSQL", "connection"],
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
        "should_find": False  # Should NOT find - not in docs
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
        "expected_keywords": ["90 days"],
        "should_find": True
    },
    {
        "query": "How do I configure webhooks?",
        "expected_keywords": ["webhook", "Settings"],
        "should_find": True
    },
]

def run_retrieval_eval():
    print("🧪 Production RAG Retrieval Evaluation\n")
    
    # Load FAISS index
    index_path = "rag/faiss_index/index.faiss"
    docs_path = "rag/faiss_index/docs.pkl"
    
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        print("❌ FAISS index not found. Run: python rag/ingest_docs.py")
        return
    
    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    results = {
        "total_queries": len(BENCHMARK_QUERIES),
        "correct_retrievals": 0,
        "correct_abstentions": 0,
        "queries": []
    }
    
    for idx, test_case in enumerate(BENCHMARK_QUERIES, 1):
        query = test_case["query"]
        should_find = test_case["should_find"]
        expected_keywords = test_case["expected_keywords"]
        
        # Retrieve top-3 documents
        query_emb = embedder.encode([query])
        distances, indices = index.search(query_emb, k=3)
        retrieved_docs = [documents[i] for i in indices[0] if i < len(documents)]
        
        # Check if keywords present in retrieved docs
        combined_text = " ".join(retrieved_docs).lower()
        has_keywords = all(kw.lower() in combined_text for kw in expected_keywords)
        
        # Retrieval success if: (should_find and has_keywords) or (not should_find and no keywords)
        success = has_keywords == should_find
        
        if should_find and success:
            results["correct_retrievals"] += 1
        elif not should_find and success:
            results["correct_abstentions"] += 1
        
        status = "✅" if success else "❌"
        print(f"{status} Q{idx}: {query[:55]}...")
        print(f"   Expected: {'Find' if should_find else 'Abstain'} | Result: {'Found' if has_keywords else 'Not found'}")
        print(f"   Top doc: {retrieved_docs[0][:80]}...")
        print()
        
        results["queries"].append({
            "query": query,
            "should_find": should_find,
            "found": has_keywords,
            "success": success,
            "top_doc": retrieved_docs[0][:100]
        })
    
    print("=" * 70)
    print("📊 RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Queries: {results['total_queries']}")
    print(f"✅ Correct Retrievals: {results['correct_retrievals']}")
    print(f"✅ Correct Abstentions: {results['correct_abstentions']}")
    accuracy = (results['correct_retrievals'] + results['correct_abstentions']) / results['total_queries']
    print(f"📈 Accuracy: {accuracy*100:.1f}%")
    print(f"📚 Indexed Documents: {len(documents)}")
    print("=" * 70)
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to benchmark_results.json\n")
    
    return results

if __name__ == "__main__":
    run_retrieval_eval()
