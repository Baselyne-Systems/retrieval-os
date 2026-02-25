"""Attaches the request_id and plan context to the active OTel span.

FastAPIInstrumentor creates the root span per request. This middleware
enriches it with retrieval-os-specific attributes that are not part of
the standard HTTP semantic conventions.
"""

from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        span = trace.get_current_span()

        if span.is_recording():
            if hasattr(request.state, "request_id"):
                span.set_attribute("retrieval_os.request_id", request.state.request_id)

        response: Response = await call_next(request)  # type: ignore[operator]
        return response
