import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from inference.run_lora_inference import generate_text

# Use lightweight RAG (no LLM model loading) if RAG_LIGHTWEIGHT=true
if os.getenv("RAG_LIGHTWEIGHT", "false").lower() == "true":
    from rag.rag_inference_lightweight import run_rag_pipeline_lightweight as run_rag_pipeline
else:
    from rag.rag_inference import run_rag_pipeline


# =========================
# Config
# =========================


@dataclass(frozen=True)
class AppConfig:
    """Application configuration loaded from environment variables."""

    api_key: str | None
    rate_limit: str
    cors_allow_origins: list[str]
    log_level: str
    request_timeout_s: float
    inference_workers: int
    inference_queue_max: int


def _split_origins(raw: str) -> list[str]:
    """Parse CORS origins from a comma-separated env var."""

    if raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def load_config() -> AppConfig:
    """Load and normalize configuration from environment variables."""

    return AppConfig(
        api_key=os.getenv("API_KEY"),
        rate_limit=os.getenv("RATE_LIMIT", "60/minute"),
        cors_allow_origins=_split_origins(os.getenv("CORS_ALLOW_ORIGINS", "*")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "20")),
        inference_workers=int(os.getenv("INFERENCE_WORKERS", "2")),
        inference_queue_max=int(os.getenv("INFERENCE_QUEUE_MAX", "128")),
    )


config = load_config()
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)


# =========================
# Observability
# =========================


class MetricsCollector:
    """In-memory metrics store for latency and abstention stats."""

    def __init__(self) -> None:
        self.latencies: dict[str, list[float]] = {}
        self.hallucinations: dict[str, int] = {"total": 0, "prevented": 0}
        self.abstentions = 0

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        if endpoint not in self.latencies:
            self.latencies[endpoint] = []
        self.latencies[endpoint].append(latency_ms)

    def record_hallucination_prevented(self) -> None:
        self.hallucinations["prevented"] += 1
        self.hallucinations["total"] += 1

    def record_hallucination_total(self) -> None:
        self.hallucinations["total"] += 1

    def record_abstention(self) -> None:
        self.abstentions += 1

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "latencies": {},
            "hallucination_rate": 0.0,
            "abstentions": self.abstentions,
        }

        for endpoint, times in self.latencies.items():
            if times:
                stats["latencies"][endpoint] = {
                    "avg_ms": round(sum(times) / len(times), 2),
                    "min_ms": round(min(times), 2),
                    "max_ms": round(max(times), 2),
                    "count": len(times),
                }

        if self.hallucinations["total"] > 0:
            prevented_rate = self.hallucinations["prevented"] / self.hallucinations["total"]
            stats["hallucination_rate"] = round(prevented_rate, 4)

        return stats


metrics = MetricsCollector()


@contextmanager
def measure_latency(endpoint: str) -> Any:
    """Record latency in milliseconds for an endpoint-scoped block."""

    start = time.perf_counter()
    try:
        yield
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record_latency(endpoint, latency_ms)


def _log_event(event: str, request: Request, **fields: Any) -> None:
    """Log a structured JSON event with request context."""

    payload = {
        "event": event,
        "method": request.method,
        "path": request.url.path,
        "request_id": getattr(request.state, "request_id", None),
        **fields,
    }
    logger.info(json.dumps(payload))


def _ensure_request_id(request: Request) -> str:
    """Ensure a request ID exists on the request state."""

    if not getattr(request.state, "request_id", None):
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    return request.state.request_id


PROM_HTTP_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["path", "status"],
)
PROM_HTTP_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "status"],
)
PROM_INFERENCE_QUEUE = Gauge(
    "inference_queue_size",
    "Inference queue size",
)


class InferenceQueueFull(Exception):
    """Raised when the inference queue is at capacity."""


@dataclass(frozen=True)
class InferenceJob:
    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future


class InferenceQueue:
    """Async inference queue with bounded concurrency."""

    def __init__(self, workers: int, maxsize: int) -> None:
        self._queue: asyncio.Queue[InferenceJob | None] = asyncio.Queue(maxsize=maxsize)
        self._workers: list[asyncio.Task] = []
        self._worker_count = max(1, workers)

    async def start(self) -> None:
        if self._workers:
            return
        for idx in range(self._worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(idx)))

    async def stop(self) -> None:
        for _ in range(len(self._workers)):
            await self._queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = InferenceJob(func=func, args=args, kwargs=kwargs, future=future)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise InferenceQueueFull("Inference queue is full") from exc
        return await future

    def size(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                return
            try:
                result = await anyio.to_thread.run_sync(
                    job.func,
                    *job.args,
                    cancellable=True,
                    **job.kwargs,
                )
                if not job.future.cancelled():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.cancelled():
                    job.future.set_exception(exc)


# =========================
# App setup
# =========================


app = FastAPI(title="Hallucination-Aware RAG API")

limiter = Limiter(key_func=get_remote_address, default_limits=[config.rate_limit])
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    app.state.inference_queue = InferenceQueue(
        workers=config.inference_workers,
        maxsize=config.inference_queue_max,
    )
    await app.state.inference_queue.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    queue = getattr(app.state, "inference_queue", None)
    if queue:
        await queue.stop()


# =========================
# Models
# =========================


class QueryRequest(BaseModel):
    """User request to run a retrieval-augmented query."""

    query: str = Field(..., min_length=1, max_length=2000)
    use_rag: bool = True


class CitationSpan(BaseModel):
    """Character span in the answer with a source document index."""

    start: int
    end: int
    doc_index: int
    snippet: str


class QueryResponse(BaseModel):
    """Response for a RAG-backed query."""

    query: str
    answer: str
    used_rag: bool
    retrieved_documents: list[str]
    citations: list[CitationSpan] = Field(default_factory=list)
    context: str | None = None
    grounded: bool | None = None
    error: str | None = None


class GenerateRequest(BaseModel):
    """User request to generate text from the base model."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(120, ge=1, le=512)


class GenerateResponse(BaseModel):
    """Response for a text-generation request."""

    output: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class MetricsResponse(BaseModel):
    """Metrics snapshot response."""

    latencies: dict[str, dict[str, float | int]]
    hallucination_rate: float
    abstentions: int


# =========================
# Middleware
# =========================


@app.middleware("http")
async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    """Attach a request ID, log latency, and return responses with trace headers."""

    request_id = _ensure_request_id(request)
    start = time.perf_counter()

    response: Response | None = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        path = request.url.path
        PROM_HTTP_LATENCY.labels(path=path, status=str(status_code)).observe(latency_ms / 1000)
        PROM_HTTP_COUNT.labels(path=path, status=str(status_code)).inc()
        queue = getattr(app.state, "inference_queue", None)
        if queue:
            PROM_INFERENCE_QUEUE.set(queue.size())
        _log_event(
            "request.completed",
            request,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
        )


@app.middleware("http")
async def api_key_guard(request: Request, call_next: Callable) -> JSONResponse:
    """Validate API key for protected routes."""

    _ensure_request_id(request)

    if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc", "/metrics/prometheus"}:
        return await call_next(request)

    if config.api_key:
        provided = request.headers.get("x-api-key")
        if not provided or not secrets.compare_digest(provided, config.api_key):
            _log_event("auth.failed", request)
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Attach request IDs to HTTP errors."""

    request_id = _ensure_request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers={"x-request-id": request_id} if request_id else None,
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit errors with structured logging."""

    request_id = _ensure_request_id(request)
    logger.exception(
        "Rate limit exceeded",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "request_id": request_id},
        headers={"x-request-id": request_id} if request_id else None,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions without leaking internal details."""

    request_id = _ensure_request_id(request)
    logger.exception(
        "Unhandled exception",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={"x-request-id": request_id} if request_id else None,
    )


# =========================
# Core services
# =========================


async def _run_in_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking function in a worker thread with a timeout."""

    async with asyncio.timeout(config.request_timeout_s):
        try:
            queue: InferenceQueue = app.state.inference_queue
            return await queue.submit(func, *args, **kwargs)
        except InferenceQueueFull as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


async def run_rag(query: str, use_rag: bool, request: Request) -> QueryResponse:
    """Execute the RAG pipeline with timeout and structured errors."""

    try:
        result = await _run_in_thread(run_rag_pipeline, query, use_rag)
    except asyncio.TimeoutError:
        logger.exception("RAG pipeline timed out", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=504, detail="RAG pipeline timed out")
    except FileNotFoundError:
        logger.exception("RAG resources unavailable", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=503, detail="RAG resources unavailable")
    except ValueError:
        logger.exception("Invalid RAG request", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=400, detail="Invalid request")
    except RuntimeError:
        logger.exception("RAG pipeline failed", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=503, detail="RAG pipeline failed")

    response = QueryResponse(
        query=result.get("query", query),
        answer=result.get("answer", ""),
        used_rag=result.get("used_rag", use_rag),
        retrieved_documents=result.get("retrieved_documents", []),
        citations=result.get("citations", []),
        context=result.get("context"),
        grounded=result.get("grounded"),
        error=result.get("error"),
    )

    if response.answer == "Not found in retrieved documents":
        metrics.record_abstention()

    return response


async def run_generation(prompt: str, max_new_tokens: int, request: Request) -> GenerateResponse:
    """Generate text with timeout and structured errors."""

    try:
        output = await _run_in_thread(generate_text, prompt, max_new_tokens=max_new_tokens)
    except asyncio.TimeoutError:
        logger.exception("Generation timed out", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=504, detail="Generation timed out")
    except ValueError:
        logger.exception("Invalid generation request", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=400, detail="Invalid request")
    except RuntimeError:
        logger.exception("Generation failed", extra={"request_id": request.state.request_id})
        raise HTTPException(status_code=503, detail="Generation failed")

    return GenerateResponse(output=output)


# =========================
# Routes
# =========================


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""

    return HealthResponse(status="ok")


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Return in-memory metrics snapshot."""

    stats = metrics.get_stats()
    return MetricsResponse(
        latencies=stats["latencies"],
        hallucination_rate=stats["hallucination_rate"],
        abstentions=stats["abstentions"],
    )


@app.get("/metrics/prometheus")
def get_prometheus_metrics() -> Response:
    """Expose Prometheus metrics."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(config.rate_limit)
async def query_llm(request: Request, req: QueryRequest) -> QueryResponse:
    """Run a RAG-backed query with latency tracking."""

    with measure_latency("/query"):
        return await run_rag(req.query, req.use_rag, request)


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit(config.rate_limit)
async def generate_llm(request: Request, req: GenerateRequest) -> GenerateResponse:
    """Generate a completion from the base model with latency tracking."""

    with measure_latency("/generate"):
        return await run_generation(req.prompt, req.max_new_tokens, request)