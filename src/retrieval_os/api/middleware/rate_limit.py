"""Redis sliding-window rate limiter middleware.

Algorithm (per-tenant, 60-second window):
  1. ZREMRANGEBYSCORE — evict timestamps older than (now - window_seconds)
  2. ZADD              — record the current request timestamp
  3. ZCARD             — count requests in the window
  4. EXPIRE            — set TTL to 2× window so idle keys expire automatically

If Redis is unavailable the middleware fails open (allows the request) and logs
a warning, prioritising availability over strict enforcement.

Rate limiting is skipped when:
  - ``settings.rate_limit_enabled = False``
  - The path starts with /health or /metrics
  - The request carries no ``tenant_id`` (unauthenticated when auth is off)
"""

from __future__ import annotations

import logging
import time

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from retrieval_os.core.config import settings
from retrieval_os.core.redis_client import get_redis

log = logging.getLogger(__name__)

_EXEMPT_PREFIXES = ("/health", "/metrics")
_WINDOW_SECONDS = 60


async def check_rate_limit(tenant_id: str, max_rpm: int) -> bool:
    """Sliding-window rate check.

    Returns:
        True if the request is within the limit, False if it should be rejected.
    """
    redis = await get_redis()
    key = f"ros:rate:{tenant_id}"
    now = time.time()
    window_start = now - _WINDOW_SECONDS

    pipe = redis.pipeline(transaction=False)
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, _WINDOW_SECONDS * 2)
    results = await pipe.execute()

    request_count: int = results[2]
    return request_count <= max_rpm


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)  # type: ignore[operator]

        path = request.url.path
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)  # type: ignore[operator]

        tenant_id: str | None = getattr(request.state, "tenant_id", None)
        if tenant_id is None:
            # Auth disabled or route exempt → no tenant context → skip.
            return await call_next(request)  # type: ignore[operator]

        max_rpm: int = getattr(
            request.state, "max_rpm", settings.rate_limit_default_rpm
        )

        try:
            allowed = await check_rate_limit(tenant_id, max_rpm)
        except Exception:
            log.warning(
                "rate_limit.redis_error",
                extra={"tenant_id": tenant_id},
                exc_info=True,
            )
            # Fail open on Redis errors.
            return await call_next(request)  # type: ignore[operator]

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": (
                        f"Rate limit exceeded. Maximum {max_rpm} requests per minute."
                    ),
                },
            )

        response: Response = await call_next(request)  # type: ignore[operator]
        return response
