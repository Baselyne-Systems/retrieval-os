"""SQLAlchemy ORM model for the Evaluation Engine domain.

An EvalJob represents a batch evaluation run against a specific plan version.
The runner embeds each query in the dataset, searches the plan's index,
and computes Recall@k, MRR, and NDCG@k against ground-truth relevant IDs.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from retrieval_os.core.database import Base
from retrieval_os.core.ids import uuid7


class EvalJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvalJob(Base):
    """A batch evaluation job for a specific plan version.

    Lifecycle: QUEUED → RUNNING → COMPLETED | FAILED

    Dataset format (S3 JSONL, one record per line):
      {"query": "...", "relevant_ids": ["id1", "id2"],
       "relevant_scores": {"id1": 1.0, "id2": 0.5}}

    relevant_scores defaults to 1.0 for all relevant_ids if omitted.
    """

    __tablename__ = "eval_jobs"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid7())
    )
    plan_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    plan_version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=EvalJobStatus.QUEUED, index=True
    )
    dataset_uri: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=10)

    # Retrieval quality metrics (null until COMPLETED)
    recall_at_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_3: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_10: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg_at_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg_at_10: Mapped[float | None] = mapped_column(Float, nullable=True)

    # LLM-as-judge quality score 1–5 (stub, populated in Phase 8)
    context_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Job stats
    total_queries: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failed_queries: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Regression vs previous completed job for the same plan
    regression_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # [{metric, prev_value, curr_value, drop_pct}, ...]
    regression_detail: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
