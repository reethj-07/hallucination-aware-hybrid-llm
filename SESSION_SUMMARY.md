# Production RAG System - Session Summary

## ✅ Current Status: OPERATIONAL

The hallucination-aware RAG system is **running and tested**. All core components are functional.

---

## 📊 System Architecture

### Core Components
1. **API Service** (`api/main.py`) - FastAPI with async queue, rate limiting, structured logging
2. **RAG Pipeline** (`rag/rag_inference_lightweight.py`) - Fast retrieval + template generation
3. **FAISS Index** (`rag/faiss_index/`) - 41 documents indexed, 62KB index file
4. **Evaluation Framework** - EM/F1/precision metrics on 15 QA pairs

### Key Optimization: Lightweight Mode
- **RAG_LIGHTWEIGHT=true** disables expensive LLM model loading
- Uses SentenceTransformer embeddings + template-based answer generation
- **Response latency**: ~20-3500ms depending on embedder load

---

## 📈 Test Results

### Retrieval Quality Evaluation
- **Success Rate**: 86.7% (13/15 queries correct)
- **Avg Precision**: 1.00 (keywords correctly retrieved)
- **Avg Recall**: 0.21 (coverage of keyword space)
- **Failures**: Q14, Q15 (intentional out-of-domain tests—system falsely retrieves content)

**Insight**: System retrieves relevant content but struggles with abstention on unrelated queries.

### Live API Testing
| Query | Response | Status |
|-------|----------|--------|
| "What is the rate limit for Standard tier API?" | "Rate Limiting: Standard tier allows 1000 requests per minute..." | ✅ Correct |
| "What encryption is used at rest and in transit?" | "...AES-256 at-rest using TLS 1.3..." | ✅ Correct |

---

## 🚀 How to Run

### Start the API (Lightweight Mode - Recommended)
```bash
cd /workspaces/hallucination-aware-hybrid-llm
RAG_LIGHTWEIGHT=true RAG_RERANK=false python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Query the API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the rate limit for Standard tier API?"}'
```

### Run Retrieval Evaluation
```bash
RAG_RERANK=false python -m scripts.evaluate_rag_retrieval
```

### Run Full RAG Pipeline (Requires ~8GB RAM)
```bash
REQUEST_TIMEOUT_S=120 RAG_RERANK=false python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 📁 Key Files

| File | Purpose | Status |
|------|---------|--------|
| [api/main.py](api/main.py) | FastAPI server with async queue | ✅ Working |
| [rag/rag_inference_lightweight.py](rag/rag_inference_lightweight.py) | Fast RAG without LLM model | ✅ Working |
| [rag/faiss_index/index.faiss](rag/faiss_index/index.faiss) | FAISS vector index | ✅ 62KB, 41 docs |
| [scripts/evaluate_rag_retrieval.py](scripts/evaluate_rag_retrieval.py) | Retrieval quality eval | ✅ Working |
| [data/eval/qa_pairs.jsonl](data/eval/qa_pairs.jsonl) | Test dataset | ✅ 15 QA pairs |

---

## 🔧 Configuration Options

Set via environment variables:

```bash
# API Timeouts
REQUEST_TIMEOUT_S=120          # Request timeout (default 20)

# RAG Behavior
RAG_LIGHTWEIGHT=true           # Use lightweight mode (skip LLM loading)
RAG_RERANK=false               # Skip cross-encoder reranking
RAG_MAX_CONTEXT_CHARS=4000     # Max context length
RAG_TOP_K=3                    # Top K docs to return
RAG_SKIP_CHUNKING=true         # Skip document chunking (for speed)

# API Settings
API_KEY=your_key               # Require API key authentication
RATE_LIMIT=60/minute           # Rate limit
CORS_ALLOW_ORIGINS=*           # CORS origins
```

---

## 🎯 Next Steps (Optional)

### 1. Full RAG with Phi-3 (Requires More Memory)
If you have 16GB+ RAM, run with full generation:
```bash
REQUEST_TIMEOUT_S=180 python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# First query will take ~60-90s to load model, subsequent queries fast
```

### 2. Improve Abstention (Out-of-Domain Detection)
- Add similarity threshold: if max score < 0.5, abstain
- Fine-tune embedder on domain-specific data
- Add explicit "not in corpus" detection in template

### 3. Optimize for Production
- Enable reranking once memory available: `RAG_RERANK=true`
- Add LRU cache for embeddings
- Implement batch query processing
- Set up Prometheus metrics scraping (`/metrics/prometheus`)

### 4. Load Testing
```bash
python -m locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 5
```

---

## 💾 Disk Usage

```
rag/faiss_index/
  ├── index.faiss      62KB   (vector index)
  └── docs.pkl         11KB   (document metadata)
                       ────
                       73KB   Total
```

---

## ⚡ Performance Baseline

| Operation | Time | Notes |
|-----------|------|-------|
| Embedder load | ~2-3s | One-time cache load |
| Query embedding | ~20-50ms | Per query |
| FAISS search | <1ms | Top-10 retrieval |
| Template generation | <1ms | No LLM call |
| **Total latency** | **20-3500ms** | Depends on cache |

First query with cold embedder cache: ~3-4s. Subsequent queries: ~20-50ms.

---

## 📝 Session Achievements

✅ Built production-grade RAG API with async concurrency
✅ Added FAISS retrieval with normalized embeddings  
✅ Created fast lightweight pipeline (skips expensive model loading)
✅ Implemented structured logging with request IDs
✅ Added retrieval quality evaluation (86.7% success rate)
✅ Set up Docker-compose for multi-service deployment
✅ Configured environment-based behavior switching
✅ Created regression test dataset (15 QA pairs)

---

## 🐛 Known Limitations

1. **Memory**: Full Phi-3 model requires 8GB+ RAM; lightweight mode avoids this
2. **Abstention**: System retrieves content even for out-of-domain queries (Q14, Q15 failures)
3. **First-query latency**: Embedder loads on first request (~3-4s)
4. **Template generation**: Naive sentence extraction; doesn't match full LLM reasoning

---

## 📞 Quick Troubleshooting

**Q: API crashes with OOM**
A: Use `RAG_LIGHTWEIGHT=true` to avoid loading Phi-3 model

**Q: Queries timeout**
A: Increase `REQUEST_TIMEOUT_S=120` and enable lightweight mode

**Q: Retrieval is poor**
A: Check index: `ls -lh rag/faiss_index/` (should show docs.pkl + index.faiss)

**Q: Want full LLM generation**
A: Remove `RAG_LIGHTWEIGHT=true` and ensure 8GB+ RAM available

---

## 🎓 System Architecture Diagram

```
Client Request
    ↓
FastAPI /query endpoint
    ↓
Structured Logging Middleware
    ↓
Rate Limiter + Auth Guard
    ↓
Async Inference Queue
    ↓
RAG Pipeline:
  ├─ Load FAISS index (cached)
  ├─ Embed query (SentenceTransformer, cached)
  ├─ Retrieve top-K documents (< 1ms)
  ├─ Rank if enabled (cross-encoder, optional)
  ├─ Budget context to 4000 chars
  ├─ [LIGHTWEIGHT] Template extraction
  └─ [FULL] Phi-3 generation (60-90s first call)
    ↓
Citation generation (optional)
    ↓
Hallucination guard (verify grounding)
    ↓
JSON response to client
```

---

**Status**: 🟢 READY FOR TESTING

All core components are operational. System demonstrates competent retrieval even in resource-constrained environment.
