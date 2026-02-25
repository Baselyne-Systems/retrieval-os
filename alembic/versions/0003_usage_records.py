"""Usage records table for the serving path.

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "usage_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("plan_name", sa.String(255), nullable=False),
        sa.Column("plan_version", sa.Integer, nullable=False),
        sa.Column("query_chars", sa.Integer, nullable=False),
        sa.Column("result_count", sa.Integer, nullable=False),
        sa.Column("cache_hit", sa.Boolean, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_usage_records_plan_name",
        "usage_records",
        ["plan_name", "created_at"],
    )
    op.create_index(
        "idx_usage_records_created_at",
        "usage_records",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("usage_records")
