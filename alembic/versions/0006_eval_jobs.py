"""Evaluation jobs table.

Revision ID: 0006
Revises: 0005
Create Date: 2026-02-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "eval_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("plan_name", sa.String(255), nullable=False),
        sa.Column("plan_version", sa.Integer, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="QUEUED"),
        sa.Column("dataset_uri", sa.Text, nullable=False),
        sa.Column("top_k", sa.Integer, nullable=False, server_default="10"),
        # Retrieval quality metrics
        sa.Column("recall_at_1", sa.Float, nullable=True),
        sa.Column("recall_at_3", sa.Float, nullable=True),
        sa.Column("recall_at_5", sa.Float, nullable=True),
        sa.Column("recall_at_10", sa.Float, nullable=True),
        sa.Column("mrr", sa.Float, nullable=True),
        sa.Column("ndcg_at_5", sa.Float, nullable=True),
        sa.Column("ndcg_at_10", sa.Float, nullable=True),
        sa.Column("context_quality_score", sa.Float, nullable=True),
        # Job stats
        sa.Column("total_queries", sa.Integer, nullable=True),
        sa.Column("failed_queries", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        # Regression detection
        sa.Column("regression_detected", sa.Boolean, nullable=True),
        sa.Column("regression_detail", postgresql.JSONB, nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=False),
    )
    op.create_index("idx_eval_jobs_plan_name", "eval_jobs", ["plan_name"])
    op.create_index("idx_eval_jobs_status", "eval_jobs", ["status"])
    op.create_index("idx_eval_jobs_created_at", "eval_jobs", ["created_at"])
    op.create_index(
        "idx_eval_jobs_plan_completed",
        "eval_jobs",
        ["plan_name", "completed_at"],
    )


def downgrade() -> None:
    op.drop_table("eval_jobs")
