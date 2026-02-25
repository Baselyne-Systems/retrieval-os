"""FastAPI application factory with lifespan and middleware stack."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from retrieval_os.api.background import (
    cost_aggregator,
    eval_job_runner,
    rollback_watchdog,
    rollout_stepper,
)
from retrieval_os.api.health import router as health_router
from retrieval_os.api.middleware.request_id import RequestIDMiddleware
from retrieval_os.api.middleware.telemetry import TelemetryMiddleware
from retrieval_os.core.config import settings
from retrieval_os.core.database import check_db_connection, engine
from retrieval_os.core.exceptions import RetrievalOSError
from retrieval_os.core.redis_client import check_redis_connection, close_redis
from retrieval_os.core.s3_client import ensure_bucket_exists
from retrieval_os.core.telemetry import setup_telemetry, shutdown_telemetry

# ── Structured logging ─────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(message)s",
)

log = structlog.get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_telemetry(
        app_name=settings.otel_service_name,
        environment=settings.environment,
        otel_endpoint=settings.otel_endpoint,
        enabled=settings.otel_enabled,
    )

    # Verify critical dependencies
    db_ok = await check_db_connection()
    redis_ok = await check_redis_connection()

    if not db_ok:
        log.error("startup.dependency_check_failed", dependency="postgres")
    if not redis_ok:
        log.error("startup.dependency_check_failed", dependency="redis")

    try:
        await ensure_bucket_exists()
    except Exception:
        log.warning("startup.s3_bucket_ensure_failed", bucket=settings.s3_bucket_name)

    log.info(
        "startup.complete",
        environment=settings.environment,
        version=settings.app_version,
        postgres=db_ok,
        redis=redis_ok,
    )

    # ── Background tasks ──────────────────────────────────────────────────────
    bg_tasks = [
        asyncio.create_task(rollback_watchdog(), name="rollback_watchdog"),
        asyncio.create_task(rollout_stepper(), name="rollout_stepper"),
        asyncio.create_task(eval_job_runner(), name="eval_job_runner"),
        asyncio.create_task(cost_aggregator(), name="cost_aggregator"),
    ]

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("shutdown.started")

    for task in bg_tasks:
        task.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)

    await close_redis()
    await engine.dispose()
    shutdown_telemetry()

    log.info("shutdown.complete")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Retrieval-OS",
        description="Multimodal Retrieval-Native Inference Runtime",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # Applied outermost-first (last added = outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Exception handlers ─────────────────────────────────────────────────────
    @app.exception_handler(RetrievalOSError)
    async def handle_retrieval_os_error(
        request: Request, exc: RetrievalOSError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "detail": exc.detail,
            },
        )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(health_router)
    # Phase 2+: plans, deployments, serving, lineage, evaluation, intelligence

    # ── OTel auto-instrumentation ──────────────────────────────────────────────
    FastAPIInstrumentor.instrument_app(app)

    return app


app = create_app()
