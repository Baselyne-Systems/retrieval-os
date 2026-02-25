"""Deployments table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-02-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("plan_name", sa.String(255), nullable=False),
        sa.Column("plan_version", sa.Integer, nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="PENDING"),
        sa.Column("traffic_weight", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("rollout_step_percent", sa.Float, nullable=True),
        sa.Column("rollout_step_interval_seconds", sa.Integer, nullable=True),
        sa.Column("rollback_recall_threshold", sa.Float, nullable=True),
        sa.Column("rollback_error_rate_threshold", sa.Float, nullable=True),
        sa.Column("change_note", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rollback_reason", sa.Text, nullable=True),
    )
    op.create_index(
        "idx_deployments_plan_name",
        "deployments",
        ["plan_name"],
    )
    op.create_index(
        "idx_deployments_status",
        "deployments",
        ["plan_name", "status"],
    )
    op.create_index(
        "idx_deployments_created_at",
        "deployments",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("deployments")
