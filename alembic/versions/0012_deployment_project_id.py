"""Add project_id FK to deployments.

Revision ID: 0012
Revises: 0011
Create Date: 2026-02-26

Changes:
- Add project_id (UUID FK → projects.id) to deployments
- Backfill project_id via JOIN on plan_name
- Make non-nullable and add index
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add nullable first so the backfill can run before we enforce NOT NULL
    op.add_column(
        "deployments",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    # Backfill from projects via plan_name
    op.execute(
        """
        UPDATE deployments d
        SET project_id = p.id
        FROM projects p
        WHERE p.name = d.plan_name
        """
    )

    # Make non-nullable and add FK + index
    op.alter_column("deployments", "project_id", nullable=False)
    op.create_foreign_key(
        "fk_deployments_project_id",
        "deployments",
        "projects",
        ["project_id"],
        ["id"],
    )
    op.create_index("idx_deployments_project_id", "deployments", ["project_id"])


def downgrade() -> None:
    op.drop_index("idx_deployments_project_id", "deployments")
    op.drop_constraint("fk_deployments_project_id", "deployments", type_="foreignkey")
    op.drop_column("deployments", "project_id")
