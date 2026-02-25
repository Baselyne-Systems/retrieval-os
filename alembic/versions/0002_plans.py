"""Plans domain — retrieval_plans and plan_versions tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrieval_plans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("is_archived", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(255), nullable=False),
    )
    op.create_index("idx_retrieval_plans_name", "retrieval_plans", ["name"], unique=True)
    op.create_index(
        "idx_retrieval_plans_created_at",
        "retrieval_plans",
        ["created_at"],
    )

    op.create_table(
        "plan_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "plan_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval_plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default="false"),

        # Embedding config
        sa.Column("embedding_provider", sa.String(100), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer, nullable=False),
        sa.Column(
            "modalities",
            postgresql.ARRAY(sa.String),
            nullable=False,
        ),
        sa.Column("embedding_batch_size", sa.Integer, nullable=False, server_default="32"),
        sa.Column("embedding_normalize", sa.Boolean, nullable=False, server_default="true"),

        # Index config
        sa.Column("index_backend", sa.String(100), nullable=False),
        sa.Column("index_collection", sa.String(255), nullable=False),
        sa.Column("distance_metric", sa.String(50), nullable=False),
        sa.Column("quantization", sa.String(50), nullable=True),

        # Retrieval config
        sa.Column("top_k", sa.Integer, nullable=False),
        sa.Column("rerank_top_k", sa.Integer, nullable=True),
        sa.Column("reranker", sa.String(255), nullable=True),
        sa.Column("hybrid_alpha", sa.Float, nullable=True),

        # Filter config
        sa.Column("metadata_filters", postgresql.JSONB, nullable=True),
        sa.Column("tenant_isolation_field", sa.String(255), nullable=True),

        # Cost config
        sa.Column("cache_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("cache_ttl_seconds", sa.Integer, nullable=False, server_default="3600"),
        sa.Column("max_tokens_per_query", sa.Integer, nullable=True),

        # Governance
        sa.Column("change_comment", sa.Text, nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),

        sa.UniqueConstraint("plan_id", "version", name="uq_plan_version"),
        sa.UniqueConstraint("plan_id", "config_hash", name="uq_plan_config_hash"),
    )
    op.create_index("idx_plan_versions_plan_id", "plan_versions", ["plan_id"])
    op.create_index(
        "idx_plan_versions_is_current",
        "plan_versions",
        ["plan_id", "is_current"],
    )


def downgrade() -> None:
    op.drop_table("plan_versions")
    op.drop_table("retrieval_plans")
