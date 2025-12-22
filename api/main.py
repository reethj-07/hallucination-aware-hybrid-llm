from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag_inference import run_rag_pipeline
from inference.run_lora_inference import generate_text

app = FastAPI(title="Hallucination-Aware RAG API")


class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/query")
def query_llm(req: QueryRequest):
    return run_rag_pipeline(req.query, req.use_rag)


@app.post("/generate")
def generate_llm(req: GenerateRequest):
    return {"output": generate_text(req.prompt)}
