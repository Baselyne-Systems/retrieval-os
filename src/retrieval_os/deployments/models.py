"""SQLAlchemy ORM models for the Deployments domain."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from retrieval_os.core.database import Base
from retrieval_os.core.ids import uuid7


class DeploymentStatus(StrEnum):
    PENDING = "PENDING"
    ROLLING_OUT = "ROLLING_OUT"
    ACTIVE = "ACTIVE"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


class Deployment(Base):
    """A deployment binds an index config to live traffic with its own search config.

    Invariants enforced at the service layer:
    - At most one deployment per project in status ACTIVE or ROLLING_OUT.
    - traffic_weight is in [0.0, 1.0].
    - rollout_step_percent is in (0, 100].
    """

    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid7()))
    project_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True
    )

    # Index config reference
    index_config_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("index_configs.id"), nullable=False
    )
    index_config_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Search config (runtime, does not affect index) ─────────────────────────
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, server_default="10")
    rerank_top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reranker: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hybrid_alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_filters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    tenant_isolation_field: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cache_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    cache_ttl_seconds: Mapped[int] = mapped_column(Integer, nullable=False, server_default="3600")
    max_tokens_per_query: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DeploymentStatus.PENDING.value
    )

    # Traffic control
    traffic_weight: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.0")
    rollout_step_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    rollout_step_interval_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Rollback thresholds
    rollback_recall_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    rollback_error_rate_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Auto-eval: if set, an eval job is automatically queued when this deployment goes ACTIVE
    eval_dataset_uri: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Governance
    change_note: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rollback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def is_live(self) -> bool:
        return self.status in (
            DeploymentStatus.ACTIVE.value,
            DeploymentStatus.ROLLING_OUT.value,
        )
