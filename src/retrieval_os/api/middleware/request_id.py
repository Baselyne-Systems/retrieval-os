"""Injects a UUIDv7 request ID into every request and response.

If the caller provides an X-Request-ID header we echo it back unchanged,
allowing end-to-end correlation across services. Otherwise we generate a
fresh UUIDv7 (time-ordered, suitable as a trace correlation key).
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from retrieval_os.core.ids import uuid7_str

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid7_str()
        request.state.request_id = request_id

        response: Response = await call_next(request)  # type: ignore[operator]
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
