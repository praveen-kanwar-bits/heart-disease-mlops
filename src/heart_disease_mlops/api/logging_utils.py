from __future__ import annotations

import json
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Request


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


async def log_request(request: Request, call_next):
    started = perf_counter()
    request_id = str(uuid4())
    response = await call_next(request)
    payload = {
        "event": "request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }
    logging.getLogger("heart_disease_mlops.api").info(json.dumps(payload))
    response.headers["X-Request-ID"] = request_id
    return response
