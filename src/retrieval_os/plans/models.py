"""SQLAlchemy ORM models for the Retrieval Plans domain."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from retrieval_os.core.database import Base
from retrieval_os.core.ids import uuid7


class RetrievalPlan(Base):
    __tablename__ = "retrieval_plans"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    versions: Mapped[list["PlanVersion"]] = relationship(
        "PlanVersion",
        back_populates="plan",
        lazy="selectin",
        order_by="PlanVersion.version",
    )

    @property
    def current_version(self) -> "PlanVersion | None":
        return next((v for v in self.versions if v.is_current), None)


class PlanVersion(Base):
    __tablename__ = "plan_versions"
    __table_args__ = (
        UniqueConstraint("plan_id", "version", name="uq_plan_version"),
        UniqueConstraint("plan_id", "config_hash", name="uq_plan_config_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid7)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("retrieval_plans.id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Embedding config ───────────────────────────────────────────────────────
    embedding_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    modalities: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    embedding_batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=32)
    embedding_normalize: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # ── Index config ───────────────────────────────────────────────────────────
    index_backend: Mapped[str] = mapped_column(String(100), nullable=False)
    index_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(50), nullable=False)
    quantization: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # ── Retrieval config ───────────────────────────────────────────────────────
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    rerank_top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reranker: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hybrid_alpha: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Filter config ──────────────────────────────────────────────────────────
    metadata_filters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    tenant_isolation_field: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # ── Cost config ────────────────────────────────────────────────────────────
    cache_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    cache_ttl_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=3600)
    max_tokens_per_query: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Governance ─────────────────────────────────────────────────────────────
    change_comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    plan: Mapped["RetrievalPlan"] = relationship("RetrievalPlan", back_populates="versions")
