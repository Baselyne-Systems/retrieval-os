"""SQLAlchemy ORM models for the Projects domain."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from retrieval_os.core.database import Base
from retrieval_os.core.ids import uuid7


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    index_configs: Mapped[list["IndexConfig"]] = relationship(
        "IndexConfig",
        back_populates="project",
        lazy="selectin",
        order_by="IndexConfig.version",
    )

    @property
    def current_index_config(self) -> "IndexConfig | None":
        return next((v for v in self.index_configs if v.is_current), None)


class IndexConfig(Base):
    __tablename__ = "index_configs"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_index_config_version"),
        UniqueConstraint("project_id", "config_hash", name="uq_index_config_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid7)
    project_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False
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

    # ── Governance ─────────────────────────────────────────────────────────────
    change_comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    project: Mapped["Project"] = relationship("Project", back_populates="index_configs")
