# rag/ingest_docs.py
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

DOCS_PATH = Path("data/rag_docs")
INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", "rag/faiss_index/index.faiss"))
STORE_PATH = Path(os.getenv("RAG_DOCS_PATH", "rag/faiss_index/docs.pkl"))
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SKIP_CHUNKING = os.getenv("RAG_SKIP_CHUNKING", "false").lower() == "true"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SIZE_CHARS = 2048
CHUNK_OVERLAP_CHARS = 256


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split documents into overlapping chunks.
    Each chunk inherits source metadata from its parent document.
    Returns list of dicts with keys: 'text', 'source', 'chunk_index', 'total_chunks'.
    """
    if RecursiveCharacterTextSplitter is None:
        chunked: list[dict] = []
        step = max(1, CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS)
        for doc in docs:
            text = str(doc["text"])
            splits = [text[i : i + CHUNK_SIZE_CHARS] for i in range(0, len(text), step)]
            for idx, split in enumerate(splits):
                chunked.append(
                    {
                        "text": split,
                        "source": doc.get("source", "unknown"),
                        "chunk_index": idx,
                        "total_chunks": len(splits),
                    }
                )
        return chunked

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunked = []
    for doc in docs:
        splits = splitter.split_text(str(doc["text"]))
        for idx, split in enumerate(splits):
            chunked.append(
                {
                    "text": split,
                    "source": doc.get("source", "unknown"),
                    "chunk_index": idx,
                    "total_chunks": len(splits),
                }
            )
    return chunked


def load_raw_docs() -> list[dict]:
    docs: list[dict] = []
    for file_path in sorted(DOCS_PATH.glob("*")):
        if file_path.is_file():
            docs.append({"text": file_path.read_text(encoding="utf-8"), "source": file_path.name})
    return docs


def build_index(force_reingest: bool = False) -> int:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    if force_reingest:
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if STORE_PATH.exists():
            STORE_PATH.unlink()

    raw_docs = load_raw_docs()
    if not raw_docs:
        print("No documents found in data/rag_docs")
        raise SystemExit(1)

    documents = raw_docs if SKIP_CHUNKING else chunk_documents(raw_docs)
    texts = [str(doc["text"]) for doc in documents]

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with STORE_PATH.open("wb") as handle:
        pickle.dump(documents, handle)

    print(
        f"Indexed {len(documents)} chunks from {len(raw_docs)} source documents "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, skip_chunking={SKIP_CHUNKING})"
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-reingest", action="store_true")
    args = parser.parse_args()
    build_index(force_reingest=args.force_reingest)


if __name__ == "__main__":
    main()
