import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
INDEX_PATH = "rag/faiss_index/index.faiss"
DOCS_PATH = "rag/faiss_index/docs.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 🔥 LLM INFERENCE ENDPOINT (LOCAL / COLAB / HF)
LLM_ENDPOINT = "http://host.docker.internal:9000/generate"

# ---------------- LOAD INDEX ----------------
index = faiss.read_index(INDEX_PATH)
print("✅ FAISS index loaded")

with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- RAG PIPELINE ----------------
def run_rag_pipeline(query: str, use_rag: bool = True):
    if use_rag:
        query_embedding = embedder.encode([query])
        _, I = index.search(query_embedding, k=3)
        context = "\n".join([documents[i] for i in I[0]])
    else:
        context = ""

    prompt = f"""
You are a senior machine learning engineer answering a technical interview question.

STRICT RULES:
- Use ONLY facts from the context
- Answer in ONE concise technical paragraph
- Do NOT repeat the question
- If the answer is not present in the context, reply exactly:
  "Not found in retrieved documents"

Context:
{context}

Question:
{query}

Final Answer:
"""

    response = requests.post(
        LLM_ENDPOINT,
        json={"prompt": prompt},
        timeout=60
    )

    return {
        "query": query,
        "answer": response.json()["output"],
        "used_rag": use_rag
    }
