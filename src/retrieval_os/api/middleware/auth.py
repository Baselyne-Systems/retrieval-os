"""API key authentication middleware — Phase 8.

Currently a pass-through stub. Phase 8 activates bcrypt-hashed API key
verification per tenant from the api_keys table.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Paths that never require authentication
_PUBLIC_PATHS = {"/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: object) -> Response:
        # Phase 8: extract X-API-Key header, hash with bcrypt, compare against
        # api_keys table. Return 401 if missing or invalid.
        # For now, pass through all requests.
        response: Response = await call_next(request)  # type: ignore[operator]
        return response
