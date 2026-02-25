"""SQLAlchemy ORM models for the Cost Intelligence domain.

ModelPricing holds embedding cost rates per provider/model.
CostEntry holds hourly aggregated cost windows derived from usage_records.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from retrieval_os.core.database import Base
from retrieval_os.core.ids import uuid7


class ModelPricing(Base):
    """Cost rate for a specific embedding provider/model pair.

    cost_per_1k_tokens is for input tokens (embeddings have no output).
    Local/self-hosted models (sentence_transformers) have cost = 0.0.
    valid_until = None means the pricing is current.
    """

    __tablename__ = "model_pricing"
    __table_args__ = (UniqueConstraint("provider", "model", "valid_from", name="uq_model_pricing"),)

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid7())
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    cost_per_1k_tokens: Mapped[float] = mapped_column(Float, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class CostEntry(Base):
    """Hourly cost aggregation for a (plan, version, provider, model) tuple.

    Populated by the cost_aggregator background loop. Idempotent — re-running
    the aggregator over the same window updates existing rows via upsert.

    token_count is estimated as total_query_chars // 4 (≈ chars per token).
    estimated_cost_usd = token_count / 1000 * cost_per_1k_tokens.
    """

    __tablename__ = "cost_entries"
    __table_args__ = (
        UniqueConstraint(
            "plan_name",
            "plan_version",
            "window_start",
            "provider",
            "model",
            name="uq_cost_entry_window",
        ),
    )

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid7())
    )
    plan_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # UTC hour boundary, e.g. 2026-02-25 14:00:00+00
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    total_queries: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False)
    # Estimated tokens from query_chars // 4
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
