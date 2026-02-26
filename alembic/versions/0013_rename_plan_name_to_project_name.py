"""Rename plan_name → project_name across deployments, ingestion_jobs, eval_jobs.

Revision ID: 0013
Revises: 0012
Create Date: 2026-02-26

Changes:
- deployments.plan_name → project_name  (+ rename indexes)
- ingestion_jobs.plan_name → project_name  (+ rename index)
- eval_jobs.plan_name → project_name  (+ rename indexes)
- eval_jobs.plan_version → index_config_version
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── deployments ────────────────────────────────────────────────────────────
    # The composite status index references plan_name; drop and recreate.
    op.drop_index("idx_deployments_status", table_name="deployments")
    op.drop_index("idx_deployments_plan_name", table_name="deployments")

    op.execute("ALTER TABLE deployments RENAME COLUMN plan_name TO project_name")

    op.create_index("idx_deployments_project_name", "deployments", ["project_name"])
    op.create_index(
        "idx_deployments_status", "deployments", ["project_name", "status"]
    )

    # ── ingestion_jobs ─────────────────────────────────────────────────────────
    op.drop_index("idx_ingestion_jobs_plan_name", table_name="ingestion_jobs")

    op.execute("ALTER TABLE ingestion_jobs RENAME COLUMN plan_name TO project_name")

    op.create_index("idx_ingestion_jobs_project_name", "ingestion_jobs", ["project_name"])

    # ── eval_jobs ──────────────────────────────────────────────────────────────
    op.drop_index("idx_eval_jobs_plan_completed", table_name="eval_jobs")
    op.drop_index("idx_eval_jobs_plan_name", table_name="eval_jobs")

    op.execute("ALTER TABLE eval_jobs RENAME COLUMN plan_name TO project_name")
    op.execute("ALTER TABLE eval_jobs RENAME COLUMN plan_version TO index_config_version")

    op.create_index("idx_eval_jobs_project_name", "eval_jobs", ["project_name"])
    op.create_index(
        "idx_eval_jobs_project_completed", "eval_jobs", ["project_name", "completed_at"]
    )


def downgrade() -> None:
    # ── eval_jobs ──────────────────────────────────────────────────────────────
    op.drop_index("idx_eval_jobs_project_completed", table_name="eval_jobs")
    op.drop_index("idx_eval_jobs_project_name", table_name="eval_jobs")

    op.execute("ALTER TABLE eval_jobs RENAME COLUMN index_config_version TO plan_version")
    op.execute("ALTER TABLE eval_jobs RENAME COLUMN project_name TO plan_name")

    op.create_index("idx_eval_jobs_plan_name", "eval_jobs", ["plan_name"])
    op.create_index(
        "idx_eval_jobs_plan_completed", "eval_jobs", ["plan_name", "completed_at"]
    )

    # ── ingestion_jobs ─────────────────────────────────────────────────────────
    op.drop_index("idx_ingestion_jobs_project_name", table_name="ingestion_jobs")

    op.execute("ALTER TABLE ingestion_jobs RENAME COLUMN project_name TO plan_name")

    op.create_index("idx_ingestion_jobs_plan_name", "ingestion_jobs", ["plan_name"])

    # ── deployments ────────────────────────────────────────────────────────────
    op.drop_index("idx_deployments_status", table_name="deployments")
    op.drop_index("idx_deployments_project_name", table_name="deployments")

    op.execute("ALTER TABLE deployments RENAME COLUMN project_name TO plan_name")

    op.create_index("idx_deployments_plan_name", "deployments", ["plan_name"])
    op.create_index("idx_deployments_status", "deployments", ["plan_name", "status"])
