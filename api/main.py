from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag_inference import run_rag_pipeline

app = FastAPI(title="Hallucination-Aware RAG API")

class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True

@app.post("/query")
def query_llm(req: QueryRequest):
    return run_rag_pipeline(
        query=req.query,
        use_rag=req.use_rag
    )
