# rag/ingest_docs.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/rag_docs"
INDEX_PATH = "rag/faiss.index"
STORE_PATH = "rag/docs.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
for file in os.listdir(DOCS_PATH):
    with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
        documents.append(f.read())

embeddings = embedder.encode(documents, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(STORE_PATH, "wb") as f:
    pickle.dump(documents, f)

print("✅ RAG documents indexed successfully")
