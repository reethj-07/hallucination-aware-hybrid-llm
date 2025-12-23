import faiss
import pickle
from sentence_transformers import SentenceTransformer
from inference.run_lora_inference import generate_text

INDEX_PATH = "rag/faiss_index/index.faiss"
DOCS_PATH = "rag/faiss_index/docs.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Load FAISS + docs
index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer(EMBED_MODEL)


def run_rag_pipeline(query: str, use_rag: bool = True):
    retrieved_docs = []

    if use_rag:
        query_embedding = embedder.encode([query])
        _, I = index.search(query_embedding, k=TOP_K)
        retrieved_docs = [documents[i] for i in I[0]]
        context = "\n".join(retrieved_docs)
    else:
        context = ""

    prompt = f"""
You are a senior machine learning engineer answering a technical question.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT use prior knowledge
- Do NOT repeat the question
- If the answer is not present in the context, reply EXACTLY:
  "Not found in retrieved documents"

Context:
{context}

Question:
{query}

Final Answer:
"""

    answer = generate_text(prompt).strip()

    # 🔒 Hallucination Guard
    if use_rag:
        context_lower = context.lower()
        answer_lower = answer.lower()

        supported = any(
            token in context_lower
            for token in answer_lower.split()
            if len(token) > 5
        )

        if not supported:
            answer = "Not found in retrieved documents"

    return {
        "query": query,
        "answer": answer,
        "used_rag": use_rag,
        "retrieved_documents": retrieved_docs
    }
