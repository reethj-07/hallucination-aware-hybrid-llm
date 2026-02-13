# rag/ingest_docs.py
import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/rag_docs"
INDEX_PATH = "rag/faiss_index/index.faiss"
STORE_PATH = "rag/faiss_index/docs.pkl"

CHUNK_SIZE_WORDS = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "500"))
CHUNK_OVERLAP_WORDS = int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", "0"))
SKIP_CHUNKING = os.getenv("RAG_SKIP_CHUNKING", "true").lower() == "true"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > 0 else end
    return chunks


# Create directory if it doesn't exist
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL)

documents: list[dict[str, str | int]] = []
for file in sorted(os.listdir(DOCS_PATH)):
    filepath = os.path.join(DOCS_PATH, file)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = [text] if SKIP_CHUNKING else _chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for idx, chunk in enumerate(chunks):
            documents.append({"text": chunk, "source": file, "chunk_id": idx})

if not documents:
    print("❌ No documents found in data/rag_docs")
    raise SystemExit(1)

texts = [doc["text"] for doc in documents]
embeddings = embedder.encode(texts, normalize_embeddings=True)
print(f"✅ Encoded {len(texts)} documents")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(STORE_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"✅ RAG documents indexed successfully ({len(documents)} chunks)")
