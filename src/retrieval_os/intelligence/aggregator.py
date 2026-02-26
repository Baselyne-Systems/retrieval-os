"""Cost aggregation logic for the Cost Intelligence domain.

The aggregator reads usage_records, groups them into 1-hour windows, looks up
the embedding cost for each plan version, and upserts cost_entries.

It processes the last `lookback_hours` complete UTC hours on every run, making
it naturally idempotent via the upsert. If the aggregator was stopped for a
day, it will catch up on the next run (subject to lookback_hours).
"""

from __future__ import annotations

import logging
import textwrap
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.ids import uuid7
from retrieval_os.intelligence.repository import intel_repo

log = logging.getLogger(__name__)

# Chars-per-token approximation (conservative estimate)
_CHARS_PER_TOKEN = 4


async def aggregate_usage_costs(
    session: AsyncSession,
    lookback_hours: int = 48,
) -> int:
    """Aggregate usage_records into cost_entries for complete 1-hour windows.

    Only processes windows that have ended (i.e. up to the start of the current
    hour). Re-running over the same window is safe — the upsert keeps the latest
    aggregated values.

    Returns the number of cost_entry rows upserted.
    """
    now = datetime.now(UTC)
    # Truncate to the current hour boundary — we only aggregate COMPLETE hours
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    since = current_hour - timedelta(hours=lookback_hours)

    # One query to get all (plan_name, plan_version, hour) buckets
    usage_sql = textwrap.dedent("""
        SELECT
            u.plan_name,
            u.plan_version,
            date_trunc('hour', u.created_at) AS window_start,
            COUNT(*)                          AS total_queries,
            SUM(CASE WHEN u.cache_hit THEN 1 ELSE 0 END) AS cache_hits,
            SUM(u.query_chars)                AS total_chars
        FROM usage_records u
        WHERE u.created_at >= :since
          AND u.created_at <  :until
        GROUP BY u.plan_name, u.plan_version, date_trunc('hour', u.created_at)
        ORDER BY window_start
    """)
    result = await session.execute(
        sa.text(usage_sql),
        {"since": since, "until": current_hour},
    )
    rows = result.fetchall()

    if not rows:
        return 0

    # Pre-load pricing cache to avoid N+1 queries
    pricing_cache: dict[tuple[str, str], float] = {}

    upserted = 0
    for row in rows:
        plan_name: str = row.plan_name
        plan_version: int = row.plan_version
        window_start: datetime = row.window_start.replace(tzinfo=UTC)
        window_end = window_start + timedelta(hours=1)
        total_queries: int = row.total_queries
        cache_hits: int = row.cache_hits
        total_chars: int = row.total_chars or 0

        # Load plan version embedding config
        provider, model = await _get_plan_embedding(session, plan_name, plan_version)
        if provider is None:
            # Plan version no longer exists — skip
            continue

        # Look up pricing (cached)
        cache_key = (provider, model)
        if cache_key not in pricing_cache:
            pricing = await intel_repo.get_pricing(session, provider, model)
            pricing_cache[cache_key] = pricing.cost_per_1k_tokens if pricing else 0.0

        cost_per_1k = pricing_cache[cache_key]
        token_count = total_chars // _CHARS_PER_TOKEN
        estimated_cost_usd = (token_count / 1000.0) * cost_per_1k

        entry_now = datetime.now(UTC)
        await intel_repo.upsert_cost_entry(
            session,
            id=str(uuid7()),
            plan_name=plan_name,
            plan_version=plan_version,
            window_start=window_start,
            window_end=window_end,
            provider=provider,
            model=model,
            total_queries=total_queries,
            cache_hits=cache_hits,
            token_count=token_count,
            estimated_cost_usd=estimated_cost_usd,
            now=entry_now,
        )
        upserted += 1

    log.info(
        "cost_aggregator.run_complete",
        extra={
            "windows_upserted": upserted,
            "lookback_hours": lookback_hours,
            "since": since.isoformat(),
            "until": current_hour.isoformat(),
        },
    )
    return upserted


async def _get_plan_embedding(
    session: AsyncSession, plan_name: str, plan_version: int
) -> tuple[str | None, str | None]:
    """Return (provider, model) for a plan version, or (None, None) if missing."""
    from sqlalchemy import select

    from retrieval_os.plans.models import IndexConfig, Project

    result = await session.execute(
        select(IndexConfig.embedding_provider, IndexConfig.embedding_model)
        .join(Project, IndexConfig.project_id == Project.id)
        .where(Project.name == plan_name, IndexConfig.version == plan_version)
    )
    row = result.one_or_none()
    if row is None:
        return None, None
    return row.embedding_provider, row.embedding_model
