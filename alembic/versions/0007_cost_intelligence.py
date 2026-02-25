"""Cost intelligence tables: model_pricing and cost_entries.

Revision ID: 0007
Revises: 0006
Create Date: 2026-02-25
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SEED_TS = datetime(2026, 2, 25, 0, 0, 0, tzinfo=UTC)


def upgrade() -> None:
    # ── model_pricing ─────────────────────────────────────────────────────────
    model_pricing = op.create_table(
        "model_pricing",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("cost_per_1k_tokens", sa.Float, nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("provider", "model", "valid_from", name="uq_model_pricing"),
    )
    op.create_index("idx_model_pricing_provider", "model_pricing", ["provider"])

    # Seed known embedding model prices (USD per 1,000 tokens, as of 2026-02)
    op.bulk_insert(
        model_pricing,
        [
            {
                "id": "01951c4b-0000-7000-8000-000000000001",
                "provider": "openai",
                "model": "text-embedding-3-small",
                "cost_per_1k_tokens": 0.00002,
                "valid_from": _SEED_TS,
                "valid_until": None,
                "created_at": _SEED_TS,
            },
            {
                "id": "01951c4b-0000-7000-8000-000000000002",
                "provider": "openai",
                "model": "text-embedding-3-large",
                "cost_per_1k_tokens": 0.00013,
                "valid_from": _SEED_TS,
                "valid_until": None,
                "created_at": _SEED_TS,
            },
            {
                "id": "01951c4b-0000-7000-8000-000000000003",
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "cost_per_1k_tokens": 0.0001,
                "valid_from": _SEED_TS,
                "valid_until": None,
                "created_at": _SEED_TS,
            },
            {
                "id": "01951c4b-0000-7000-8000-000000000004",
                "provider": "cohere",
                "model": "embed-english-v3.0",
                "cost_per_1k_tokens": 0.0001,
                "valid_from": _SEED_TS,
                "valid_until": None,
                "created_at": _SEED_TS,
            },
            {
                "id": "01951c4b-0000-7000-8000-000000000005",
                "provider": "cohere",
                "model": "embed-multilingual-v3.0",
                "cost_per_1k_tokens": 0.0001,
                "valid_from": _SEED_TS,
                "valid_until": None,
                "created_at": _SEED_TS,
            },
            {
                "id": "01951c4b-0000-7000-8000-000000000006",
                "provider": "sentence_transformers",
                "model": "__default__",
                "cost_per_1k_tokens": 0.0,
                "valid_from": _SEED_TS,
                "valid_until": None,
                "created_at": _SEED_TS,
            },
        ],
    )

    # ── cost_entries ──────────────────────────────────────────────────────────
    op.create_table(
        "cost_entries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("plan_name", sa.String(255), nullable=False),
        sa.Column("plan_version", sa.Integer, nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("total_queries", sa.Integer, nullable=False),
        sa.Column("cache_hits", sa.Integer, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("estimated_cost_usd", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "plan_name", "plan_version", "window_start", "provider", "model",
            name="uq_cost_entry_window",
        ),
    )
    op.create_index("idx_cost_entries_plan_name", "cost_entries", ["plan_name"])
    op.create_index("idx_cost_entries_window_start", "cost_entries", ["window_start"])
    op.create_index(
        "idx_cost_entries_plan_window",
        "cost_entries",
        ["plan_name", "window_start"],
    )


def downgrade() -> None:
    op.drop_table("cost_entries")
    op.drop_table("model_pricing")
