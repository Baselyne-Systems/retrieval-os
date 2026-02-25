"""ORM model for IngestionJob."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from retrieval_os.core.database import Base


class IngestionJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IngestionJob(Base):
    """A batch document ingestion job: chunk → embed → upsert → lineage."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    plan_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)

    # Source — one of source_uri (S3) or document_payload (inline JSON list)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON list of {id, content, metadata} when submitted inline
    document_payload: Mapped[list | None] = mapped_column(JSON, nullable=True)

    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=512)
    overlap: Mapped[int] = mapped_column(Integer, nullable=False, default=64)

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=IngestionJobStatus.QUEUED.value, index=True
    )

    # Progress counters — populated once the job starts / completes
    total_docs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    indexed_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failed_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
