from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

http_request_latency = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["path", "status"],
)
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "status"],
)
inference_queue_size = Gauge(
    "inference_queue_size",
    "Inference queue size",
)

rag_faithfulness_score = Histogram(
    "rag_faithfulness_score",
    "NLI faithfulness score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
rag_abstentions_total = Counter(
    "rag_abstentions_total",
    "Number of times the guard triggered abstention",
    labelnames=["guard_method"],
)
rag_retrieval_success_total = Counter(
    "rag_retrieval_success_total",
    "Queries where at least one document was retrieved",
    labelnames=["model_mode"],
)
embedding_cache_hits_total = Counter(
    "embedding_cache_hits_total",
    "Embedding LRU cache hits",
)


def setup_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_middleware(request, call_next):
        import time

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        status = str(response.status_code)
        http_request_latency.labels(path=path, status=status).observe(elapsed)
        http_requests_total.labels(path=path, status=status).inc()
        return response

    @app.get("/metrics/prometheus")
    def get_prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
