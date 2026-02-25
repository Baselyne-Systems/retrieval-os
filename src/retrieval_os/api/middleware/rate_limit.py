"""Redis sliding-window rate limiter — Phase 8.

Currently a pass-through stub. Phase 8 activates per-(tenant, plan_name)
rate limiting using a Redis sorted set sliding window algorithm.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        # Phase 8: extract tenant_id from request.state, check Redis sliding
        # window counter, return 429 if limit exceeded.
        response: Response = await call_next(request)  # type: ignore[operator]
        return response
