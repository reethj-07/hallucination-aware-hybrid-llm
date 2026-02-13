# 🧠 Hallucination-Aware Hybrid LLM System
### Production-Grade RAG with Phi-3 and FAISS Retrieval

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.111.0-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/Phi--3-Mini-yellow)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![Retrieval](https://img.shields.io/badge/FAISS-1.8-blue)](https://github.com/facebookresearch/faiss)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](./SESSION_SUMMARY.md)

---

## 🚀 Overview

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** that prevents hallucinations through **strict context-grounding**. Responses are retrieved from a FAISS vector index and verified against source documents before returning to users.

**Key Innovation**: Two deployment modes:
- **Lightweight Mode** (`RAG_LIGHTWEIGHT=true`) — Fast retrieval + template generation (20-50ms)
- **Full Mode** — Phi-3 LoRA with cross-encoder reranking (for 8GB+ RAM systems)

**Current Status**: 🟢 Operational with 86.7% retrieval accuracy on test queries

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
4. **Two Deployment Modes**:
   - **Lightweight** (`RAG_LIGHTWEIGHT=true`): ~20-50ms latency, 200MB memory
   - **Full** (default): 100-500ms latency, 8GB+ memory

---

##  Hallucination Control Logic

The system enforces correctness using **two independent and complementary safeguards** to minimize hallucinations.

### 1️⃣ Retrieval Constraint (Knowledge Grounding)

- User queries are embedded using a SentenceTransformer
- Top-K relevant documents are retrieved via FAISS
- **Only retrieved documents** are passed to the language model as context
- No external or prior model knowledge is allowed during generation

If no relevant document is retrieved, the system forces abstention.

---

### 2️⃣ Prompt-Level Generation Constraints

The language model is instructed with **strict generation rules**:

text
- Answer ONLY using the provided context
- Do NOT use prior knowledge
- Do NOT repeat the question
- If the answer is not present in the context, reply EXACTLY:
  "Not found in retrieved documents"


---

##  SECTION 2 — RAG Pipeline (Step-by-Step)

This shows *engineering clarity*.

markdown
##  Retrieval-Augmented Generation (RAG) Pipeline

The RAG pipeline follows a deterministic, auditable sequence:

1. **Query Encoding**  
   The user query is converted into a dense vector embedding.

2. **Document Retrieval**  
   FAISS performs similarity search over the indexed document corpus.

3. **Context Assembly**  
   The top-K retrieved documents are concatenated into a single context block.

4. **Constrained Prompt Construction**  
   The context and query are injected into a hallucination-safe prompt template.

5. **LLM Generation**  
   A LoRA-fine-tuned Phi-3 Mini model generates the final response.

6. **Abstention Check**  
   If the answer is not grounded in context, the model explicitly refuses.

##  Inference Modes: RAG vs Non-RAG

The system supports two inference modes for comparison and evaluation:

### ❌ Non-RAG Mode
- Direct LLM inference without document retrieval
- Model may rely on internal parametric knowledge
- Susceptible to hallucinations

### ✅ RAG Mode (Default)
- Responses are grounded in retrieved documents
- Hallucination guardrails enforced
- Transparent inspection of retrieved context

This dual-mode setup highlights the **impact of retrieval grounding on factual correctness**.

## 📁 Project Structure

text
hallucination-aware-hybrid-llm/
│
├── app/                     # Streamlit UI
├── api/                     # FastAPI endpoints
├── inference/               # LoRA-based inference logic
├── models/                  # Fine-tuned LoRA adapters
├── rag/                     # Retrieval & hallucination-aware pipeline
│   └── faiss_index/         # Vector index + documents
├── training/                # QLoRA fine-tuning scripts
├── experiments/             # Jupyter notebooks
├── Dockerfile
├── requirements.txt
└── README.md


---

## SECTION 5 — Example Behavior (VERY GOOD FOR DEMOS)


##  Example Behavior

| Mode | Question | Output |
|-----|---------|--------|
| ❌ Non-RAG | What are bottlenecks of attention? | Hallucinated / unsupported |
| ✅ RAG | What is quantization? | Grounded answer + sources |
| ❌ Non-RAG | Random ML trivia | Model may hallucinate |
| ✅ RAG | Unsupported query | "Not found in retrieved documents" |


## 🛠 Tech Stack

| Component | Technology |
|---------|------------|
| LLM | Microsoft Phi-3 Mini (4k) |
| Fine-Tuning | QLoRA (PEFT) |
| Retrieval | FAISS |
| Embeddings | Sentence-Transformers |
| Backend | Python, PyTorch |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

## Future Improvements

- Confidence-based abstention scoring
- Cross-encoder reranking for improved retrieval precision
- Adaptive top-K retrieval
- Hallucination rate benchmarking
- Token-level document attribution
- Self-verification and reflection loops
