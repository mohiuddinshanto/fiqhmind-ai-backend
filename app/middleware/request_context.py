import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id, log every request with latency, and echo the id back."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        logger = structlog.get_logger("http.request")

        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 1)

        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.bind_contextvars(
            status_code=response.status_code, duration_ms=duration_ms
        )
        logger.info("request_completed")
        structlog.contextvars.clear_contextvars()
        return response
