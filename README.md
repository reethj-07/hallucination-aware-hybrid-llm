# Hallucination-Aware Hybrid LLM

Production-ready and research-grade RAG system with FastAPI, FAISS, SentenceTransformers, and Phi-3 Mini.

## Overview

This repository provides two runtime modes:
- Lightweight mode: template-based generation over retrieved chunks (fast CPU path)
- Full mode: Phi-3 Mini + LoRA generation over retrieved chunks

Core capabilities:
- Chunked FAISS indexing with metadata-rich citations
- Multi-layer hallucination guard with NLI faithfulness fallback
- Modular FastAPI app with routers, settings, middleware, and metrics
- Evaluation stack with EM/F1 + RAGAS + MLflow logging

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Build index with chunking (recommended default)

```bash
RAG_SKIP_CHUNKING=false python -m rag.ingest_docs --force-reingest
```

### 3) Start API

```bash
RAG_LIGHTWEIGHT=true python -m uvicorn api.main:app --port 8000
```

### 4) Query API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the rate limit for Standard tier?"}'
```

Response includes:
- `faithfulness_score`
- `guard_method`
- `retrieval_scores`
- `latency_ms`
- `model_mode`

## Metrics

Use the expanded evaluation pipeline:

```bash
python -m scripts.build_eval_dataset
python -m scripts.evaluate_rag --eval-set data/eval/qa_pairs_v2.jsonl
```

Tracked metrics:
- EM, F1
- avg NLI faithfulness
- RAGAS faithfulness
- RAGAS answer relevancy
- RAGAS context precision
- RAGAS context recall

Note: earlier published retrieval figures (including 86.7% from the legacy setup) are historical baseline numbers and not directly comparable to the new chunked + NLI-guard pipeline.

## Lightweight Mode Clarification

`RAG_LIGHTWEIGHT=true` is template-based answer extraction, not LLM inference. It is designed for low-latency CPU serving and regression testing.

## Quickstart for Research

### MLflow setup

```bash
set MLFLOW_TRACKING_URI=./mlruns
set MLFLOW_EXPERIMENT_NAME=rag-evals
```

### Run ablation study

```bash
python -m experiments.ablation_guard_layers --eval-set data/eval/qa_pairs_v2.jsonl
```

### Compare embedder models

```bash
set RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
python -m scripts.evaluate_rag --eval-set data/eval/qa_pairs_v2.jsonl
```

Repeat with another embedder value and compare MLflow runs.

## Architecture and Finetuning Docs

- System architecture: docs/ARCHITECTURE.md
- QLoRA guide: finetune/README.md

## Current Status

- Chunked indexing: complete
- NLI faithfulness guard: complete
- API modular refactor: complete
- Strategy-pattern pipelines: complete
- Expanded evaluation + MLflow: complete
- Guard-layer ablation framework: complete
- CI coverage gating and quality tooling: complete

## Configuration

See `.env.example` for all environment variables and safe defaults.

## Docker

```bash
docker compose build
docker compose up --wait
curl http://localhost:8000/health
```
