"""Health and infrastructure endpoints."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from retrieval_os.core.config import settings
from retrieval_os.core.database import check_db_connection
from retrieval_os.core.redis_client import check_redis_connection

router = APIRouter(tags=["infrastructure"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Always returns 200 while the process is alive."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready() -> JSONResponse:
    """
    Returns 200 when all critical dependencies (Postgres, Redis) are reachable.
    Returns 503 when any dependency is unavailable — use this as a k8s
    readinessProbe so the pod is removed from Service endpoints until ready.
    """
    checks = {
        "postgres": await check_db_connection(),
        "redis": await check_redis_connection(),
    }
    all_ok = all(checks.values())

    return JSONResponse(
        content={
            "status": "ready" if all_ok else "degraded",
            "checks": {k: "ok" if v else "error" for k, v in checks.items()},
        },
        status_code=200 if all_ok else 503,
    )


@router.get("/metrics", summary="Prometheus scrape endpoint", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/info", summary="Service metadata")
async def info() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }
