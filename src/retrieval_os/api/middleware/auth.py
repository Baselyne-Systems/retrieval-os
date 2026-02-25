"""API key authentication middleware.

When ``settings.auth_enabled = True``, every request must carry a valid API key
in the ``X-API-Key`` header (configurable via ``settings.api_key_header``).

Key lookup is O(1): the client sends the full key, we extract the prefix for an
indexed DB lookup then confirm with the SHA-256 hash.  The tenant's rate-limit
quota is attached to ``request.state.max_rpm`` for the RateLimitMiddleware.

Exempt paths (no auth required):
  /health/*, /metrics, /docs, /redoc, /openapi.json
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from retrieval_os.core.config import settings
from retrieval_os.core.database import async_session_factory
from retrieval_os.tenants.repository import get_api_key_by_prefix_and_hash

log = logging.getLogger(__name__)

_EXEMPT_PREFIXES = ("/health", "/metrics", "/docs", "/redoc", "/openapi.json")


def _extract_prefix(key: str) -> str | None:
    """Return the prefix segment of an API key, or None if malformed.

    Expected format: ``ros_<8-hex>_<rest>``  → prefix = ``ros_<8-hex>``.
    """
    if not key.startswith("ros_"):
        return None
    parts = key.split("_", 2)  # ["ros", "<8-hex>", "<rest>"]
    if len(parts) < 3:  # noqa: PLR2004
        return None
    return f"ros_{parts[1]}"


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        if not settings.auth_enabled:
            return await call_next(request)  # type: ignore[operator]

        path = request.url.path
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)  # type: ignore[operator]

        raw_key = request.headers.get(settings.api_key_header)
        if not raw_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "AUTHENTICATION_REQUIRED",
                    "message": f"Missing {settings.api_key_header} header",
                },
            )

        prefix = _extract_prefix(raw_key)
        if not prefix:
            return JSONResponse(
                status_code=401,
                content={"error": "INVALID_API_KEY", "message": "Malformed API key"},
            )

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        async with async_session_factory() as session:
            api_key = await get_api_key_by_prefix_and_hash(session, prefix, key_hash)

        if api_key is None or not api_key.is_active:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "INVALID_API_KEY",
                    "message": "Invalid or revoked API key",
                },
            )

        if api_key.expires_at and api_key.expires_at < datetime.now(UTC):
            return JSONResponse(
                status_code=401,
                content={"error": "EXPIRED_API_KEY", "message": "API key has expired"},
            )

        # Attach tenant context — used by RateLimitMiddleware and downstream handlers.
        request.state.tenant_id = api_key.tenant_id
        request.state.api_key_id = api_key.id
        # max_rpm loaded via the selectin relationship on ApiKey.tenant
        request.state.max_rpm = api_key.tenant.max_requests_per_minute

        log.debug(
            "auth.ok",
            extra={
                "tenant_id": api_key.tenant_id,
                "key_prefix": api_key.key_prefix,
            },
        )

        response: Response = await call_next(request)  # type: ignore[operator]
        return response
