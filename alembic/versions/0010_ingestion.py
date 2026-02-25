"""Document ingestion job tracking table.

Revision ID: 0010
Revises: 0009
Create Date: 2026-02-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("plan_name", sa.String(255), nullable=False),
        sa.Column("plan_version", sa.Integer, nullable=False),
        # Source: one of source_uri (S3) or document_payload (inline JSON)
        sa.Column("source_uri", sa.Text, nullable=True),
        sa.Column("document_payload", sa.JSON, nullable=True),
        sa.Column("chunk_size", sa.Integer, nullable=False, default=512),
        sa.Column("overlap", sa.Integer, nullable=False, default=64),
        sa.Column("status", sa.String(20), nullable=False, default="QUEUED"),
        # Progress counters
        sa.Column("total_docs", sa.Integer, nullable=True),
        sa.Column("total_chunks", sa.Integer, nullable=True),
        sa.Column("indexed_chunks", sa.Integer, nullable=True),
        sa.Column("failed_chunks", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
    )
    op.create_index("idx_ingestion_jobs_plan_name", "ingestion_jobs", ["plan_name"])
    op.create_index("idx_ingestion_jobs_status", "ingestion_jobs", ["status"])
    op.create_index(
        "idx_ingestion_jobs_status_created_at",
        "ingestion_jobs",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
