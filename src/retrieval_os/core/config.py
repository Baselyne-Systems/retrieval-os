from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "retrieval-os"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://retrieval_os:retrieval_os@localhost:5432/retrieval_os"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    database_echo: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 10
    redis_socket_timeout: float = 5.0

    # ── S3 / MinIO ────────────────────────────────────────────────────────────
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_bucket_name: str = "retrieval-os"
    s3_region: str = "us-east-1"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_grpc_port: int = 6334
    qdrant_http_port: int = 6333
    qdrant_api_key: str | None = None

    # ── Observability ─────────────────────────────────────────────────────────
    otel_endpoint: str = "http://localhost:4317"
    otel_enabled: bool = True
    otel_service_name: str = "retrieval-os-api"

    # ── Background task intervals ─────────────────────────────────────────────
    rollback_watchdog_interval_seconds: int = 30
    rollout_stepper_interval_seconds: int = 10
    eval_job_poll_interval_seconds: int = 5
    cost_aggregator_interval_seconds: int = 3600


settings = Settings()
