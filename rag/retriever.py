import faiss
import pickle
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    def __init__(self, index_path, docs_path):
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, k=3):
        q_emb = self.embedder.encode([query])
        _, idx = self.index.search(q_emb, k)
        return [self.docs[i] for i in idx[0]]
