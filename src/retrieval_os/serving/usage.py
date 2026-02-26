"""Usage record persistence.

Each query writes a lightweight usage_record row asynchronously via
asyncio.create_task() — the serving path does not await it.
Phase 7 (Cost Intelligence) aggregates these into cost_entries.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker

log = logging.getLogger(__name__)

# ── ORM-free insert (avoid import cycle with plans.models) ────────────────────

_usage_records = sa.table(
    "usage_records",
    sa.column("id", sa.String),
    sa.column("project_name", sa.String),
    sa.column("index_config_version", sa.Integer),
    sa.column("query_chars", sa.Integer),
    sa.column("result_count", sa.Integer),
    sa.column("cache_hit", sa.Boolean),
    sa.column("latency_ms", sa.Float),
    sa.column("created_at", sa.DateTime(timezone=True)),
)


async def _write_usage(
    session_factory: async_sessionmaker,
    *,
    record_id: str,
    project_name: str,
    index_config_version: int,
    query_chars: int,
    result_count: int,
    cache_hit: bool,
    latency_ms: float,
) -> None:
    try:
        async with session_factory() as session:
            await session.execute(
                sa.insert(_usage_records).values(
                    id=record_id,
                    project_name=project_name,
                    index_config_version=index_config_version,
                    query_chars=query_chars,
                    result_count=result_count,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
    except Exception:
        log.warning("usage.write_failed", extra={"project": project_name})


def fire_usage_record(
    session_factory: async_sessionmaker,
    *,
    record_id: str,
    project_name: str,
    index_config_version: int,
    query_chars: int,
    result_count: int,
    cache_hit: bool,
    latency_ms: float,
) -> None:
    """Schedule a usage record write without blocking the response."""
    asyncio.create_task(
        _write_usage(
            session_factory,
            record_id=record_id,
            project_name=project_name,
            index_config_version=index_config_version,
            query_chars=query_chars,
            result_count=result_count,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
        ),
        name=f"usage_write_{project_name}",
    )
