"""Webhook subscriptions for outbound event notifications.

Revision ID: 0009
Revises: 0008
Create Date: 2026-02-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "webhook_subscriptions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("url", sa.String(2048), nullable=False),
        # JSON list of event strings; empty list = all events
        sa.Column("events", sa.JSON, nullable=False),
        sa.Column("secret", sa.String(255), nullable=True),
        sa.Column("description", sa.String(1024), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_webhook_subscriptions_is_active", "webhook_subscriptions", ["is_active"])


def downgrade() -> None:
    op.drop_table("webhook_subscriptions")
