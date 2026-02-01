# rag/ingest_docs.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/rag_docs"
INDEX_PATH = "rag/faiss_index/index.faiss"
STORE_PATH = "rag/faiss_index/docs.pkl"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
for file in sorted(os.listdir(DOCS_PATH)):
    filepath = os.path.join(DOCS_PATH, file)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            documents.append(f.read())

if not documents:
    print("❌ No documents found in data/rag_docs")
    exit(1)

embeddings = embedder.encode(documents, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(STORE_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"✅ RAG documents indexed successfully ({len(documents)} documents)")
