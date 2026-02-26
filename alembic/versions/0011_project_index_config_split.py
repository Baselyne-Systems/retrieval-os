"""Project rename + IndexConfig/Deployment split.

Revision ID: 0011
Revises: 0010
Create Date: 2026-02-26

Changes:
- Rename retrieval_plans → projects
- Create index_configs table (narrowed from plan_versions — index fields only)
- Migrate plan_versions data into index_configs
- Add search config columns to deployments + backfill from plan_versions
- Add index_config_id FK to deployments + backfill
- Add index_config_id FK + rename plan_version → index_config_version in ingestion_jobs
- Drop plan_versions table
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── 1. Rename retrieval_plans → projects ──────────────────────────────────
    op.rename_table("retrieval_plans", "projects")

    # ── 2. Create index_configs table ─────────────────────────────────────────
    op.create_table(
        "index_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="false"),
        # Embedding config
        sa.Column("embedding_provider", sa.String(100), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("modalities", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column(
            "embedding_batch_size", sa.Integer(), nullable=False, server_default="32"
        ),
        sa.Column(
            "embedding_normalize", sa.Boolean(), nullable=False, server_default="true"
        ),
        # Index config
        sa.Column("index_backend", sa.String(100), nullable=False),
        sa.Column("index_collection", sa.String(255), nullable=False),
        sa.Column("distance_metric", sa.String(50), nullable=False),
        sa.Column("quantization", sa.String(50), nullable=True),
        # Governance
        sa.Column("change_comment", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.UniqueConstraint("project_id", "version", name="uq_index_config_version"),
        sa.UniqueConstraint("project_id", "config_hash", name="uq_index_config_hash"),
    )
    op.create_index("idx_index_configs_project_id", "index_configs", ["project_id"])

    # ── 3. Migrate plan_versions → index_configs ──────────────────────────────
    op.execute(
        """
        INSERT INTO index_configs (
            id, project_id, version, is_current,
            embedding_provider, embedding_model, embedding_dimensions, modalities,
            embedding_batch_size, embedding_normalize,
            index_backend, index_collection, distance_metric, quantization,
            change_comment, created_at, created_by, config_hash
        )
        SELECT
            id, plan_id, version, is_current,
            embedding_provider, embedding_model, embedding_dimensions, modalities,
            embedding_batch_size, embedding_normalize,
            index_backend, index_collection, distance_metric, quantization,
            change_comment, created_at, created_by, config_hash
        FROM plan_versions
        """
    )

    # ── 4. Add search config + FK columns to deployments (nullable initially) ─
    op.add_column(
        "deployments",
        sa.Column("index_config_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "deployments",
        sa.Column("index_config_version", sa.Integer(), nullable=True),
    )
    op.add_column("deployments", sa.Column("top_k", sa.Integer(), nullable=True))
    op.add_column("deployments", sa.Column("rerank_top_k", sa.Integer(), nullable=True))
    op.add_column("deployments", sa.Column("reranker", sa.String(255), nullable=True))
    op.add_column("deployments", sa.Column("hybrid_alpha", sa.Float(), nullable=True))
    op.add_column(
        "deployments", sa.Column("metadata_filters", postgresql.JSONB(), nullable=True)
    )
    op.add_column(
        "deployments",
        sa.Column("tenant_isolation_field", sa.String(255), nullable=True),
    )
    op.add_column("deployments", sa.Column("cache_enabled", sa.Boolean(), nullable=True))
    op.add_column(
        "deployments", sa.Column("cache_ttl_seconds", sa.Integer(), nullable=True)
    )
    op.add_column(
        "deployments", sa.Column("max_tokens_per_query", sa.Integer(), nullable=True)
    )

    # ── 5. Backfill deployments from plan_versions ────────────────────────────
    op.execute(
        """
        UPDATE deployments d
        SET
            index_config_id       = pv.id,
            index_config_version  = pv.version,
            top_k                 = pv.top_k,
            rerank_top_k          = pv.rerank_top_k,
            reranker              = pv.reranker,
            hybrid_alpha          = pv.hybrid_alpha,
            metadata_filters      = pv.metadata_filters,
            tenant_isolation_field = pv.tenant_isolation_field,
            cache_enabled         = pv.cache_enabled,
            cache_ttl_seconds     = pv.cache_ttl_seconds,
            max_tokens_per_query  = pv.max_tokens_per_query
        FROM plan_versions pv
        JOIN projects p ON pv.plan_id = p.id
        WHERE p.name = d.plan_name
          AND pv.version = d.plan_version
        """
    )

    # Set defaults for any rows that had no matching plan_version (safety net)
    op.execute(
        """
        UPDATE deployments
        SET
            top_k             = COALESCE(top_k, 10),
            cache_enabled     = COALESCE(cache_enabled, true),
            cache_ttl_seconds = COALESCE(cache_ttl_seconds, 3600)
        WHERE top_k IS NULL OR cache_enabled IS NULL OR cache_ttl_seconds IS NULL
        """
    )

    # ── 6. Make non-nullable and add FK ───────────────────────────────────────
    op.alter_column("deployments", "index_config_version", nullable=False)
    op.alter_column("deployments", "top_k", nullable=False)
    op.alter_column("deployments", "cache_enabled", nullable=False)
    op.alter_column("deployments", "cache_ttl_seconds", nullable=False)

    # Only add FK if there are rows (safe for both fresh installs and migrations)
    op.create_foreign_key(
        "fk_deployments_index_config_id",
        "deployments",
        "index_configs",
        ["index_config_id"],
        ["id"],
    )

    # ── 7. Drop plan_version column from deployments ──────────────────────────
    op.drop_column("deployments", "plan_version")

    # ── 8. Update ingestion_jobs ──────────────────────────────────────────────
    op.add_column(
        "ingestion_jobs",
        sa.Column("index_config_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("index_config_version", sa.Integer(), nullable=True),
    )

    # Backfill ingestion_jobs from plan_versions
    op.execute(
        """
        UPDATE ingestion_jobs ij
        SET
            index_config_id      = pv.id,
            index_config_version = pv.version
        FROM plan_versions pv
        JOIN projects p ON pv.plan_id = p.id
        WHERE p.name = ij.plan_name
          AND pv.version = ij.plan_version
        """
    )

    # Safety net for any unmatched rows
    op.execute(
        """
        UPDATE ingestion_jobs
        SET index_config_version = COALESCE(index_config_version, plan_version)
        WHERE index_config_version IS NULL
        """
    )

    op.alter_column("ingestion_jobs", "index_config_version", nullable=False)

    op.create_foreign_key(
        "fk_ingestion_jobs_index_config_id",
        "ingestion_jobs",
        "index_configs",
        ["index_config_id"],
        ["id"],
    )

    # Drop old plan_version column from ingestion_jobs
    op.drop_column("ingestion_jobs", "plan_version")

    # ── 9. Drop plan_versions table ───────────────────────────────────────────
    op.drop_table("plan_versions")


def downgrade() -> None:
    # Recreate plan_versions (minimal — no data recovery)
    op.create_table(
        "plan_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "plan_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("embedding_provider", sa.String(100), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("modalities", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("embedding_batch_size", sa.Integer(), nullable=False, server_default="32"),
        sa.Column("embedding_normalize", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("index_backend", sa.String(100), nullable=False),
        sa.Column("index_collection", sa.String(255), nullable=False),
        sa.Column("distance_metric", sa.String(50), nullable=False),
        sa.Column("quantization", sa.String(50), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("rerank_top_k", sa.Integer(), nullable=True),
        sa.Column("reranker", sa.String(255), nullable=True),
        sa.Column("hybrid_alpha", sa.Float(), nullable=True),
        sa.Column("metadata_filters", postgresql.JSONB(), nullable=True),
        sa.Column("tenant_isolation_field", sa.String(255), nullable=True),
        sa.Column("cache_enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("cache_ttl_seconds", sa.Integer(), nullable=False, server_default="3600"),
        sa.Column("max_tokens_per_query", sa.Integer(), nullable=True),
        sa.Column("change_comment", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
    )

    # Restore ingestion_jobs.plan_version
    op.add_column(
        "ingestion_jobs", sa.Column("plan_version", sa.Integer(), nullable=True)
    )
    op.execute(
        "UPDATE ingestion_jobs SET plan_version = index_config_version "
        "WHERE plan_version IS NULL"
    )
    op.drop_constraint(
        "fk_ingestion_jobs_index_config_id", "ingestion_jobs", type_="foreignkey"
    )
    op.drop_column("ingestion_jobs", "index_config_id")
    op.drop_column("ingestion_jobs", "index_config_version")

    # Restore deployments.plan_version + remove search config columns
    op.add_column(
        "deployments", sa.Column("plan_version", sa.Integer(), nullable=True)
    )
    op.execute(
        "UPDATE deployments SET plan_version = index_config_version "
        "WHERE plan_version IS NULL"
    )
    op.drop_constraint(
        "fk_deployments_index_config_id", "deployments", type_="foreignkey"
    )
    for col in [
        "index_config_id",
        "index_config_version",
        "top_k",
        "rerank_top_k",
        "reranker",
        "hybrid_alpha",
        "metadata_filters",
        "tenant_isolation_field",
        "cache_enabled",
        "cache_ttl_seconds",
        "max_tokens_per_query",
    ]:
        op.drop_column("deployments", col)

    op.drop_index("idx_index_configs_project_id", "index_configs")
    op.drop_table("index_configs")
    op.rename_table("projects", "retrieval_plans")
