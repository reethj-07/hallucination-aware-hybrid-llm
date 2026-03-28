# Hallucination-Aware Hybrid LLM Architecture

## 1) System Overview

```
Client
  |
  v
FastAPI (api/app.py)
  |
  +--> /query router
  |      |
  |      v
  |   Pipeline Factory (rag/pipeline.py)
  |      |
  |      +--> LightweightPipeline (template generation)
  |      |
  |      +--> FullPipeline (Phi-3 + LoRA generation)
  |             |
  |             v
  |       BaseRAGPipeline
  |        - embed_query
  |        - retrieve (FAISS)
  |        - verify_answer (NLI/token fallback)
  |
  +--> /health router

Shared Components:
- rag/shared.py: cached embedder + FAISS + docs store loaders
- rag/ingest_docs.py: chunking + embedding + index build
- rag/hallucination_guard.py: multi-layer faithfulness guard
```

## 2) Hallucination Guard Layers

### Layer 1: NLI Faithfulness (Primary)
- Model: cross-encoder/nli-deberta-v3-small
- Input:
  - premise = concatenated retrieved context
  - hypothesis = generated answer
- Output: entailment probability in [0, 1]
- Decision: abstain if score < `FAITHFULNESS_THRESHOLD` (default 0.4)

### Layer 2: Token Overlap (Fallback)
- Used when NLI model is unavailable or disabled
- Stopword-removed unigram Jaccard overlap:

$$
\text{score} = \frac{|A \cap C|}{|A \cup C|}
$$

- $A$ = answer tokens, $C$ = context tokens

### Layer 3: Abstention Trigger
- If score < threshold, return exact abstention string:
  - `Not found in retrieved documents`

## 3) Deployment Modes

### Lightweight Mode
- `RAG_LIGHTWEIGHT=true`
- Template-based generation
- Typical latency: 20-50ms
- Memory: low (CPU-friendly)
- Best for API responsiveness and local testing

### Full Mode
- `RAG_LIGHTWEIGHT=false`
- Phi-3 Mini + LoRA generation
- Typical latency: 150-700ms
- Memory: higher (benefits from GPU)
- Best for richer generation behavior

## 4) FAISS Indexing Strategy

- Source docs: `data/rag_docs/`
- Chunking:
  - chunk size: 2048 chars (~512 tokens)
  - overlap: 256 chars (~64 tokens)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Similarity metric: inner product on normalized embeddings (cosine-equivalent)
- Artifacts:
  - index: `rag/faiss_index/index.faiss`
  - metadata: `rag/faiss_index/docs.pkl`

Chunk metadata includes:
- `source`
- `chunk_index`
- `total_chunks`

## 5) Evaluation Methodology

- Dataset:
  - Existing domain pairs (`data/eval/qa_pairs.jsonl`)
  - Expanded set (`data/eval/qa_pairs_v2.jsonl`) with NQ-Open samples
- Scripts:
  - `python -m scripts.build_eval_dataset`
  - `python -m scripts.evaluate_rag --eval-set data/eval/qa_pairs_v2.jsonl`
- Metrics:
  - EM, F1
  - NLI faithfulness (pipeline score)
  - RAGAS: faithfulness, answer relevancy, context precision, context recall

## 6) Experiment Tracking (MLflow)

- Local default:
  - `MLFLOW_TRACKING_URI=./mlruns`
  - `MLFLOW_EXPERIMENT_NAME=rag-evals`
- Run:
  - `python -m scripts.evaluate_rag --eval-set data/eval/qa_pairs_v2.jsonl`
- View UI:
  - `mlflow ui --backend-store-uri ./mlruns`
  - Open `http://localhost:5000`
