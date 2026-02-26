"""Add eval_dataset_uri to deployments and duplicate_of to ingestion_jobs.

Revision ID: 0014
Revises: 0013
Create Date: 2026-02-26

Changes:
- deployments: ADD COLUMN eval_dataset_uri TEXT NULL
- ingestion_jobs: ADD COLUMN duplicate_of VARCHAR(36) NULL
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("deployments", sa.Column("eval_dataset_uri", sa.Text(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("duplicate_of", sa.String(36), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "duplicate_of")
    op.drop_column("deployments", "eval_dataset_uri")
