# 🧠 Hallucination-Aware Hybrid LLM System
### Production-Grade RAG with Phi-3 and FAISS Retrieval

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.111.0-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/Phi--3-Mini-yellow)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![Retrieval](https://img.shields.io/badge/FAISS-1.8-blue)](https://github.com/facebookresearch/faiss)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](#-quickstart)

---

## 🚀 Overview

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** that prevents hallucinations through **strict context-grounding**. Responses are retrieved from a FAISS vector index and verified against source documents before returning to users.

**Key Innovation**: Two deployment modes:
- **Lightweight Mode** (`RAG_LIGHTWEIGHT=true`) — Fast retrieval + template generation (20-50ms, 200MB memory)
- **Full Mode** — Phi-3 LoRA with cross-encoder reranking (100-500ms, 8GB+ memory)

**Current Status**: 🟢 Operational with **86.7% retrieval accuracy** on test queries

---

## ✨ Key Features

- **🔍 FAISS Vector Retrieval** — Fast, accurate document similarity search on normalized embeddings  
- **📚 Normalized Embeddings** — Inner-product similarity with L2 normalization for stable retrieval  
- **⚡ Lightweight & Full Modes** — Choose between fast template generation or full LLM reasoning  
- **🚦 Async Inference Queue** — Bounded concurrency with configurable worker threads  
- **📊 Cross-Encoder Reranking** — Optional LRU-cached reranking for improved precision  
- **🛡️ Hallucination Guards** — Multi-level safeguards: retrieval constraints + prompt engineering + token overlap verification  
- **🔐 Production Features** — Rate limiting, API key auth, structured logging with request IDs, Prometheus metrics  
- **📈 Evaluation Framework** — EM/F1/citation precision metrics on regression dataset  
- **🐳 Docker-Ready** — Separate API + UI services with docker-compose  
- **📝 Citations & Grounding** — Track which documents support each answer with span-level citations  

---

## 🏗 System Architecture

```
Client Request
    ↓
FastAPI /query endpoint
    ↓
Middleware: Auth + Rate Limiting + Logging
    ↓
Async Inference Queue (bounded concurrency)
    ↓
RAG Pipeline:
  ├─ Load FAISS index (LRU cache)
  ├─ Embed query (SentenceTransformer, cached)
  ├─ Retrieve top-K documents (< 1ms)
  ├─ Optional: Cross-encoder reranking (LRU cache)
  ├─ Budget context to MAX_CONTEXT_CHARS
  ├─ [LIGHTWEIGHT] Template-based answer extraction
  └─ [FULL] Phi-3 LoRA generation (first call: 60-90s, subsequent: 100-500ms)
    ↓
Citation generation (span-level grounding)
    ↓
Hallucination guard (token overlap verification)
    ↓
JSON response with metrics to client
```

### Key Design Decisions

1. **Normalized Embeddings**: Inner-product similarity instead of L2 distance for improved retrieval
2. **Context Budgeting**: 4000-char limit prevents token overflow in model context
3. **LRU Caching**: Models, embeddings, and reranker scores cached for 2048+ entries
4. **Two Deployment Modes**: Choose performance vs. full reasoning based on hardware

---

## 🛡️ Hallucination Prevention Strategy

The system uses **three independent safeguards** to prevent hallucinations:

### 1️⃣ Retrieval Constraint
- Query embedded using SentenceTransformer (all-MiniLM-L6-v2)
- Top-K relevant documents retrieved via FAISS inner-product search
- **Only retrieved documents** passed as context to LLM
- No parametric knowledge allowed during generation

### 2️⃣ Prompt-Level Generation Constraints
Strict instructions in system prompt:
```
- Answer ONLY using the provided context
- Do NOT use prior knowledge
- If answer not in context, respond exactly:
  "Not found in retrieved documents"
```

### 3️⃣ Token Overlap Verification
- Generated answer verified against retrieved documents
- If below threshold overlap, forces abstention
- Optional: Cross-encoder reranking validates retrieved relevance
- Citations track exact document spans used

---

## 📊 Evaluation Results

**Retrieval Quality (15 QA pairs):**
- ✅ Success Rate: **86.7%** (13/15 correct retrievals)
- ✅ Precision: **1.00** (100% of retrieved queries matched expected keywords)
- ✅ Recall: **0.21** (average keyword coverage)
- ⚠️ Known Issue: False retrieval for out-of-domain queries (Q14-Q15)

**Live API Testing:**
| Query | Response | Latency | Status |
|-------|----------|---------|--------|
| "What is the rate limit?" | "1000 requests per minute..." | 20-50ms | ✅ |
| "What encryption is used?" | "AES-256 at-rest, TLS 1.3 in transit..." | 20-50ms | ✅ |

See [SESSION_SUMMARY.md](SESSION_SUMMARY.md) for detailed metrics and test results.

---

## 🚀 Quickstart

### Prerequisites
```bash
Python 3.12+
~500MB RAM (lightweight mode) or 8GB+ RAM (full mode)
```

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build FAISS Index
```bash
# First time only (creates 41-doc index in seconds)
RAG_SKIP_CHUNKING=true RAG_RERANK=false python -m rag.ingest_docs

# Output:
# ✅ Encoded 41 documents
# ✅ RAG documents indexed successfully (41 chunks)
```

### 3. Start API (Lightweight Mode - Recommended)
```bash
RAG_LIGHTWEIGHT=true RAG_RERANK=false python -m uvicorn api.main:app --port 8000

# Server ready at http://localhost:8000
```

### 4. Test with Sample Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the rate limit for Standard tier?"}'
```

**Response** (~50ms):
```json
{
  "query": "What is the rate limit for Standard tier?",
  "answer": "Rate Limiting: Standard tier allows 1000 requests per minute...",
  "used_rag": true,
  "retrieved_documents": [
    "Rate Limiting: Standard tier allows 1000 requests per minute...",
    "..."
  ],
  "citations": []
}
```

### 5. Run Evaluation
```bash
# Fast retrieval-only evaluation (no LLM)
RAG_RERANK=false python -m scripts.evaluate_retrieval

# Output:
# Retrieval Success Rate: 86.7%
# Avg Precision: 1.00
# Results saved to retrieval_results.json
```

---

## 📁 Project Structure

```
hallucination-aware-hybrid-llm/
├── api/
│   ├── main.py                      # FastAPI server (537 lines)
│   ├── metrics.py                   # Prometheus endpoint
│   └── middleware.py                # Auth, rate limiting, logging
├── rag/
│   ├── rag_inference.py             # Full RAG + Phi-3 (268 lines)
│   ├── rag_inference_lightweight.py # Fast template-based (150 lines)
│   ├── ingest_docs.py               # Build FAISS index
│   ├── pipeline.py                  # Pipeline orchestration
│   └── faiss_index/
│       ├── index.faiss              # Vector index (62KB)
│       └── docs.pkl                 # Document metadata (11KB)
├── inference/
│   └── run_lora_inference.py        # Phi-3 + LoRA generation
├── scripts/
│   ├── evaluate_rag.py              # Full RAG evaluation
│   └── evaluate_retrieval.py        # Retrieval-only metrics
├── data/
│   ├── eval/
│   │   └── qa_pairs.jsonl           # 15 regression test QA pairs
│   ├── rag_docs/                    # 41 source documents
│   └── finetune/
├── tests/
│   ├── test_api.py                  # API tests
│   └── test_rag_pipeline.py         # RAG pipeline tests
├── app/
│   └── streamlit_app.py             # UI (optional)
├── docker-compose.yml               # Multi-service deployment
├── Dockerfile                       # API image
├── requirements.txt                 # All deps pinned
├── SESSION_SUMMARY.md               # Detailed session notes
└── README.md                        # This file
```

---

## ⚙️ Configuration

Set via environment variables:

```bash
# RAG Behavior
RAG_LIGHTWEIGHT=true              # Use fast template mode (recommended)
RAG_RERANK=false                  # Skip cross-encoder reranking
RAG_MAX_CONTEXT_CHARS=4000        # Max context length
RAG_TOP_K=3                       # Top K docs to return
RAG_SKIP_CHUNKING=true            # Skip chunking (for speed)
RAG_INDEX_PATH=rag/faiss_index/index.faiss
RAG_DOCS_PATH=rag/faiss_index/docs.pkl

# API & Server
REQUEST_TIMEOUT_S=20              # Request timeout
API_KEY=your_secret_key           # Require API key auth (optional)
RATE_LIMIT=60/minute              # Rate limit per IP
CORS_ALLOW_ORIGINS=*              # CORS allowed origins
LOG_LEVEL=INFO                    # Logging level

# Models
RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
BASE_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
LORA_PATH=models/phi3_lora_final
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI 0.111.0, Uvicorn, Starlette |
| **LLM** | Microsoft Phi-3 Mini (4k) + LoRA (PEFT 0.10.0) |
| **Retrieval** | FAISS 1.8 (inner-product) |
| **Embeddings** | SentenceTransformer 2.7.0 (all-MiniLM-L6-v2) |
| **Reranking** | CrossEncoder 2.x (optional) |
| **Async** | asyncio, anyio, threading |
| **Observability** | Prometheus 0.20.0, structured JSON logging |
| **Rate Limiting** | SlowAPI 0.1.9 |
| **UI** | Streamlit (optional) |
| **Testing** | pytest, locust |
| **Deployment** | Docker, docker-compose |

---

## 📈 Performance

| Operation | Lightweight | Full Mode | Notes |
|-----------|------------|-----------|-------|
| Cold embedder load | ~2-3s | ~2-3s | One-time per process |
| Query embedding | 20-50ms | 20-50ms | Cached |
| FAISS search | <1ms | <1ms | Top-10 retrieval |
| Reranking (if enabled) | N/A | 50-200ms | LRU cached |
| Generation | N/A | 100-500ms | Phi-3 on CPU |
| **Total per query** | **20-50ms** | **150-700ms** | First call slower |

---

## 🔄 API Endpoints

### `/query` (POST)
Retrieve documents and generate grounded response.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is X?", "use_rag": true}'
```

**Response:**
```json
{
  "query": "What is X?",
  "answer": "...",
  "used_rag": true,
  "retrieved_documents": ["...", "..."],
  "citations": [{"start": 0, "end": 10, "doc_index": 0, "snippet": "..."}]
}
```

### `/health` (GET)
Health check endpoint.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `/metrics/prometheus` (GET)
Prometheus metrics endpoint.

```bash
curl http://localhost:8000/metrics/prometheus
# http_request_latency_seconds{endpoint="/query",...} 0.052
# http_requests_total{endpoint="/query",...} 5
# inference_queue_size 0
```

---

## 🐳 Docker Deployment

### Build & Run with Docker Compose

```bash
docker compose build
docker compose up
```

**Services:**
- **API**: http://localhost:8000 (FastAPI)
- **UI**: http://localhost:8501 (Streamlit, optional)

### Environment Configuration

Create `.env` file:
```
RAG_LIGHTWEIGHT=true
RAG_RERANK=false
API_KEY=your_secret_key
RATE_LIMIT=100/minute
```

---

## 🧪 Testing

### Retrieval Evaluation (Fast, No LLM)
```bash
RAG_RERANK=false python -m scripts.evaluate_retrieval
```

### Full RAG Evaluation (Requires Phi-3)
```bash
REQUEST_TIMEOUT_S=180 python -m scripts.evaluate_rag
```

### Load Testing
```bash
python -m locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 5
# Open http://localhost:8089 in browser
```

### Unit Tests
```bash
pytest tests/
```

---

## 📝 Example Outputs

### ✅ Correct Answer (Grounded)
```
Query: "What is the rate limit for Standard tier?"
Answer: "Rate Limiting: Standard tier allows 1000 requests per minute. 
Premium tier allows 10000 requests per minute. Enterprise tier has custom limits."
Status: ✅ Correct, grounded in retrieved documents
```

### ✅ Correct Abstention (Not in Docs)
```
Query: "What quantum computing features are available?"
Answer: "Not found in retrieved documents"
Status: ✅ Correct - system recognized out-of-domain query
```

### ⚠️ False Retrieval (Known Issue)
```
Query: "Does the platform support blockchain smart contracts?"
Retrieved: [Irrelevant docs about security/compliance]
Status: ⚠️ Known limitation - insufficient out-of-domain detection
Fix: Implement similarity threshold or explicit out-of-corpus detection
```

---

## 🔮 Future Improvements

- [ ] Adaptive similarity threshold for abstention
- [ ] Fine-tuned domain-specific embedder
- [ ] Explicit "not in corpus" detection in generation
- [ ] Confidence scoring per answer
- [ ] Knowledge graph-based retrieval
- [ ] Multi-hop reasoning
- [ ] Streaming response support
- [ ] Batch query processing
- [ ] Custom LoRA adapter selection

---

## 📚 References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers](https://www.sbert.net/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [LoRA Fine-Tuning](https://github.com/microsoft/LoRA)

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.

---

## ✉️ Session Notes

For detailed implementation notes, architecture decisions, and troubleshooting, see [SESSION_SUMMARY.md](SESSION_SUMMARY.md).

**Status Update**: System is operational with 86.7% retrieval accuracy. Ready for deployment and further optimization.
