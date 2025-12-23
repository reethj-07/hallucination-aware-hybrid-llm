# 🧠 Hallucination-Aware RAG System

A production-style **Hallucination-Aware Retrieval-Augmented Generation (RAG)** system that enforces **context-grounded responses** using a LoRA-fine-tuned Phi-3 Mini language model.

This project demonstrates how to **detect and suppress hallucinations** in LLM outputs by combining retrieval-based grounding with strict inference-time constraints.

---

## 🚀 Key Features

- 🔎 **FAISS-based document retrieval** for grounding responses in external knowledge
- 🧩 **LoRA fine-tuning** of Phi-3 Mini for parameter-efficient adaptation
- 🛑 **Hallucination guardrails** that block unsupported model outputs
- 🧠 **Context-only generation enforcement** (no prior knowledge leakage)
- 📊 **Retrieved document inspection** for transparency and debugging
- 🌐 **Interactive Streamlit UI**, deployed on Hugging Face Spaces

---

## 🏗️ Architecture Overview


---

## 📂 Project Structure


---

## 🧪 Hallucination Control Logic

The system enforces correctness using **two independent safeguards**:

1. **Prompt-level constraints**
   - Model is instructed to answer *only* from retrieved context
   - Explicit refusal clause if information is missing

2. **Post-generation validation**
   - Model output is checked against retrieved documents
   - Unsupported answers are replaced with:
     ```
     Not found in retrieved documents
     ```

This ensures the model **cannot hallucinate confidently**.

---

## 🧠 Example Behavior

| Query | RAG Enabled | Output |
|-----|------------|-------|
| "What is RAG?" | ✅ | Grounded, factual answer |
| "What is quantization?" | ❌ | Free-form LLM answer |
| "Bottlenecks of attention mechanism?" | ✅ | Refusal if not in docs |

---

## 🌐 Live Demo

👉 **Hugging Face Space**:  
https://huggingface.co/spaces/attentionseeker/hallucination-aware-rag

---

## 🛠️ Tech Stack

- **LLM**: Phi-3 Mini (Microsoft)
- **Fine-tuning**: LoRA (PEFT)
- **Vector Store**: FAISS
- **Embeddings**: SentenceTransformers
- **Frontend**: Streamlit
- **Deployment**: Hugging Face Spaces

---

## 📌 Why This Project Matters

This project addresses a critical limitation of modern LLMs — **hallucinations** — by demonstrating a practical, deployable solution that balances generation power with factual correctness.

It is especially relevant for:
- Enterprise QA systems
- Technical interview assistants
- Trust-sensitive AI applications

---

## 📈 Future Improvements

- Add confidence scores based on retrieval similarity
- Highlight evidence spans used in answers
- Support multi-hop retrieval
- Introduce evaluation metrics (faithfulness, groundedness)
- GPU-accelerated inference for lower latency

---

## 📜 License

MIT
