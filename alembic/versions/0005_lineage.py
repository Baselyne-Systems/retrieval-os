"""Lineage artifacts and edges tables.

Revision ID: 0005
Revises: 0004
Create Date: 2026-02-25
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "lineage_artifacts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("artifact_type", sa.String(50), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(100), nullable=False),
        sa.Column("storage_uri", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("metadata", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
    )
    op.create_index(
        "idx_lineage_artifacts_type",
        "lineage_artifacts",
        ["artifact_type"],
    )
    op.create_index(
        "idx_lineage_artifacts_storage_uri",
        "lineage_artifacts",
        ["storage_uri"],
        unique=True,
    )
    op.create_index(
        "idx_lineage_artifacts_created_at",
        "lineage_artifacts",
        ["created_at"],
    )

    op.create_table(
        "lineage_edges",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "parent_artifact_id",
            sa.String(36),
            sa.ForeignKey("lineage_artifacts.id"),
            nullable=False,
        ),
        sa.Column(
            "child_artifact_id",
            sa.String(36),
            sa.ForeignKey("lineage_artifacts.id"),
            nullable=False,
        ),
        sa.Column("relationship_type", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.UniqueConstraint(
            "parent_artifact_id", "child_artifact_id", name="uq_lineage_edge"
        ),
    )
    op.create_index(
        "idx_lineage_edges_parent",
        "lineage_edges",
        ["parent_artifact_id"],
    )
    op.create_index(
        "idx_lineage_edges_child",
        "lineage_edges",
        ["child_artifact_id"],
    )


def downgrade() -> None:
    op.drop_table("lineage_edges")
    op.drop_table("lineage_artifacts")
