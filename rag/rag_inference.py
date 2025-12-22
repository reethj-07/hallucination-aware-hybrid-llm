import faiss
import pickle
from sentence_transformers import SentenceTransformer
from inference.run_lora_inference import generate_text

INDEX_PATH = "rag/faiss_index/index.faiss"
DOCS_PATH = "rag/faiss_index/docs.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer(EMBED_MODEL)


def run_rag_pipeline(query: str, use_rag: bool = True):
    if use_rag:
        emb = embedder.encode([query])
        _, I = index.search(emb, k=3)
        context = "\n".join(documents[i] for i in I[0])
    else:
        context = ""

    prompt = f"""
Use ONLY the context below.
If answer not present, say:
"Not found in retrieved documents"

Context:
{context}

Question:
{query}

Answer:
"""

    output = generate_text(prompt)

    return {
        "query": query,
        "answer": output,
        "used_rag": use_rag
    }
