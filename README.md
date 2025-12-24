# 🧠 Hallucination-Aware Hybrid LLM System  
### Retrieval-Augmented Generation with LoRA-Fine-Tuned Phi-3

---

## 🚀 Overview

Large Language Models (LLMs) frequently hallucinate when answering questions without sufficient grounding.  
This project implements a **hallucination-aware Retrieval-Augmented Generation (RAG) system** that enforces **context-only generation**, ensuring responses are strictly derived from retrieved documents.

The system integrates:
- **FAISS-based dense retrieval**
- **QLoRA fine-tuned Phi-3 Mini (4k)**
- **Strict prompt-level hallucination guardrails**
- **Transparent retrieved-document inspection**
- **Interactive Streamlit UI deployed on Hugging Face Spaces**

---

## ✨ Key Features

-  **FAISS-based vector retrieval** for grounding LLM responses in external knowledge  
-  **QLoRA fine-tuning of Phi-3 Mini (4k)** for parameter-efficient adaptation  
-  **Hallucination guardrails** enforcing context-only generation  
-  **Explicit abstention mechanism** when answers are not present in retrieved documents  
-  **Retrieved document inspection** for explainability and debugging  
-  **Dual inference modes**: RAG vs Non-RAG comparison  
-  **Streamlit UI deployment** on Hugging Face Spaces  

---

##  Architecture Overview

```mermaid
flowchart LR
    User --> UI[Streamlit UI]
    UI --> RAG[RAG Pipeline]
    RAG --> Embedder[SentenceTransformer]
    Embedder --> FAISS[FAISS Index]
    FAISS --> Docs[Top-K Documents]
    Docs --> Prompt[Constrained Prompt Builder]
    Prompt --> LLM[Phi-3 Mini + LoRA]
    LLM --> Answer
    Answer --> UI

---
## Hallucination Control Logic

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

```text
- Answer ONLY using the provided context
- Do NOT use prior knowledge
- Do NOT repeat the question
- If the answer is not present in the context, reply EXACTLY:
  "Not found in retrieved documents"


💡 **Why this section matters**  
Recruiters immediately see: *you understand hallucinations at a system-design level, not just prompt tricks.*

---

##  SECTION 2 — RAG Pipeline (Step-by-Step)

This shows **engineering clarity**.

```markdown
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

```text
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

```markdown
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


