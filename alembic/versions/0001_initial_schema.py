"""Initial schema — Phase 1 baseline.

Creates no tables. Later phases add domain-specific tables:
  0002_plans.py        — retrieval_plans, plan_versions
  0003_deployments.py  — deployments, rollback_events
  0004_lineage.py      — lineage_artifacts, lineage_edges, plan_version_artifacts
  0005_eval.py         — eval_jobs, eval_runs, regression_alerts

Revision ID: 0001
Revises:
Create Date: 2026-02-25
"""

from typing import Sequence, Union

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
