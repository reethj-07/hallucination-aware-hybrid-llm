import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from rag.rag_inference import run_rag_pipeline
from inference.run_lora_inference import generate_text
from api.middleware import LoggingMiddleware
from api.metrics import metrics, measure_latency

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = FastAPI(title="Hallucination-Aware RAG API")
app.add_middleware(LoggingMiddleware, logger=logger)

API_KEY = os.getenv("API_KEY")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
if CORS_ALLOW_ORIGINS == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = True


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(120, ge=1, le=512)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    return metrics.get_stats()


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)

    if API_KEY:
        provided = request.headers.get("x-api-key")
        if provided != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    return await call_next(request)


@app.post("/query")
@limiter.limit(RATE_LIMIT)
def query_llm(request: Request, req: QueryRequest):
    try:
        with measure_latency("/query"):
            result = run_rag_pipeline(req.query, req.use_rag)
            if result.get("answer") == "Not found in retrieved documents":
                metrics.record_abstention()
            return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate")
@limiter.limit(RATE_LIMIT)
def generate_llm(request: Request, req: GenerateRequest):
    try:
        with measure_latency("/generate"):
            return {"output": generate_text(req.prompt, max_new_tokens=req.max_new_tokens)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
