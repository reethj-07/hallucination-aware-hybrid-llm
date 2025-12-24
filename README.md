# 🧠 Hallucination-Aware Hybrid LLM System
### Retrieval-Augmented Generation (RAG) with LoRA-Fine-Tuned Phi-3

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Phi--3-Mini-yellow)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![Deployment](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces)

---

## 🚀 Overview

Large Language Models (LLMs) frequently hallucinate when answering questions without sufficient grounding. This project implements a **hallucination-aware Retrieval-Augmented Generation (RAG) system** that enforces **context-only generation**, ensuring responses are strictly derived from retrieved documents.

The system integrates **FAISS-based dense retrieval**, a **QLoRA fine-tuned Phi-3 Mini (4k)** model, and strict prompt-level guardrails to deliver accurate, explainable answers. It features a dual-mode inference engine (RAG vs. Non-RAG) and is deployed via an interactive Streamlit UI.

---

## ✨ Key Features

- **FAISS-based Vector Retrieval:** Grounds LLM responses in external, verifiable knowledge.
- **QLoRA Fine-Tuning:** Parameter-efficient adaptation of the Phi-3 Mini (4k) model.
- **Hallucination Guardrails:** Enforces context-only generation to prevent fabrication.
- **Abstention Mechanism:** Explicitly refuses to answer if information is missing from the context.
- **Transparent Inspection:** Users can view the exact retrieved documents used for generation.
- **Dual Inference Modes:** Compare **RAG (Grounded)** vs. **Non-RAG (Parametric)** outputs in real-time.
- **Deployment:** Hosted on Hugging Face Spaces with a Streamlit frontend.

---

## 🏗 System Architecture

The data flow ensures that the LLM is isolated from generating purely based on training weights when in RAG mode.

```mermaid
flowchart LR
    User([User]) --> UI[Streamlit UI]
    UI --> Switch{Mode?}
    
    subgraph RAG Pipeline
    Switch -- RAG --> Embedder[SentenceTransformer]
    Embedder --> FAISS[FAISS Index]
    FAISS --> Docs[Top-K Documents]
    Docs --> Prompt[Constrained Prompt Builder]
    end
    
    Switch -- Non-RAG --> RawPrompt[Standard Prompt]
    
    Prompt --> LLM[Phi-3 Mini + LoRA]
    RawPrompt --> LLM
    LLM --> Answer[Generated Response]
    Answer --> UI
