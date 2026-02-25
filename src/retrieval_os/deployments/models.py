"""SQLAlchemy ORM models for the Deployments domain."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Float, Integer, String, Text
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
    """A deployment binds a plan version to live traffic.

    Invariants enforced at the service layer:
    - At most one deployment per plan in status ACTIVE or ROLLING_OUT.
    - traffic_weight is in [0.0, 1.0].
    - rollout_step_percent is in (0, 100].
    """

    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid7())
    )
    plan_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DeploymentStatus.PENDING.value
    )

    # Traffic control
    traffic_weight: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0.0"
    )
    # Gradual rollout: percent to increment each step (None = instant full deploy)
    rollout_step_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    # How often to advance (seconds between steps)
    rollout_step_interval_seconds: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    # Rollback thresholds (optional — watchdog only triggers when set)
    rollback_recall_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    rollback_error_rate_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Governance
    change_note: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=""
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rolled_back_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rollback_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def is_live(self) -> bool:
        return self.status in (
            DeploymentStatus.ACTIVE.value,
            DeploymentStatus.ROLLING_OUT.value,
        )
