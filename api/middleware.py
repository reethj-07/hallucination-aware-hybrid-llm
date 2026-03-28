from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request
from starlette.responses import Response

from api.config import Settings

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level.upper())

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        payload = {
            "event": "request.completed",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "request_id": request_id,
        }
        logger.info(json.dumps(payload))
        response.headers["x-request-id"] = request_id
        return response
