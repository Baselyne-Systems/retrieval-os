"""Business logic for the Cost Intelligence domain."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core import metrics
from retrieval_os.core.ids import uuid7
from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.intelligence.models import ModelPricing
from retrieval_os.intelligence.recommender import PlanStats, generate_recommendations
from retrieval_os.intelligence.repository import intel_repo
from retrieval_os.intelligence.schemas import (
    AddModelPricingRequest,
    CostEntryResponse,
    CostSummaryResponse,
    CostSummaryRow,
    ModelPricingResponse,
    Recommendation,
    RecommendationsResponse,
)
from retrieval_os.plans.models import IndexConfig, Project

log = logging.getLogger(__name__)


# ── Model pricing ─────────────────────────────────────────────────────────────


async def list_model_pricing(session: AsyncSession) -> list[ModelPricingResponse]:
    entries = await intel_repo.list_pricing(session)
    return [ModelPricingResponse.model_validate(e) for e in entries]


async def add_model_pricing(
    session: AsyncSession, request: AddModelPricingRequest
) -> ModelPricingResponse:
    now = datetime.now(UTC)
    entry = ModelPricing(
        id=str(uuid7()),
        provider=request.provider,
        model=request.model,
        cost_per_1k_tokens=request.cost_per_1k_tokens,
        valid_from=now,
        valid_until=None,
        created_at=now,
    )
    entry = await intel_repo.add_pricing(session, entry)
    return ModelPricingResponse.model_validate(entry)


# ── Cost entries ──────────────────────────────────────────────────────────────


async def list_cost_entries(
    session: AsyncSession,
    project_name: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[CostEntryResponse], int]:
    if since is None:
        since = datetime.now(UTC) - timedelta(days=7)
    if until is None:
        until = datetime.now(UTC)

    entries, total = await intel_repo.list_entries(
        session,
        project_name=project_name,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    return [CostEntryResponse.model_validate(e) for e in entries], total


# ── Cost summary ──────────────────────────────────────────────────────────────


async def get_cost_summary(
    session: AsyncSession,
    since: datetime | None = None,
    until: datetime | None = None,
    project_name: str | None = None,
) -> CostSummaryResponse:
    if since is None:
        since = datetime.now(UTC) - timedelta(days=7)
    if until is None:
        until = datetime.now(UTC)

    rows = await intel_repo.get_summary(
        session, since=since, until=until, project_name=project_name
    )

    summary_rows: list[CostSummaryRow] = []
    grand_total = 0.0
    for row in rows:
        total_q = row["total_queries"] or 0
        cache_h = row["cache_hits"] or 0
        cost = float(row["estimated_cost_usd"] or 0.0)
        grand_total += cost
        summary_rows.append(
            CostSummaryRow(
                project_name=row["project_name"],
                total_queries=total_q,
                cache_hits=cache_h,
                cache_hit_rate=cache_h / total_q if total_q > 0 else 0.0,
                token_count=row["token_count"] or 0,
                estimated_cost_usd=cost,
                cost_per_query_usd=cost / total_q if total_q > 0 else 0.0,
            )
        )

    # Update Prometheus cache efficiency gauges
    for row in summary_rows:
        metrics.cache_efficiency_ratio.labels(project_name=row.project_name).set(row.cache_hit_rate)

    return CostSummaryResponse(
        since=since, until=until, rows=summary_rows, grand_total_usd=grand_total
    )


# ── Recommendations ───────────────────────────────────────────────────────────


async def get_recommendations(
    session: AsyncSession,
    project_name: str | None = None,
) -> RecommendationsResponse:
    """Compute rule-based recommendations from recent cost data and project config."""
    since = datetime.now(UTC) - timedelta(days=1)
    until = datetime.now(UTC)

    # Get cost summary for last 24 h
    summary_rows = await intel_repo.get_summary(
        session, since=since, until=until, project_name=project_name
    )
    summary_by_project = {r["project_name"]: r for r in summary_rows}

    # Get current index config for projects that have usage
    project_names = list(summary_by_project) if summary_by_project else []
    if project_name and project_name not in project_names:
        project_names.append(project_name)

    # Load active deployments with their index configs for projects that have usage
    plan_stats: list[PlanStats] = []
    if project_names:
        result = await session.execute(
            select(Deployment, IndexConfig.embedding_provider, IndexConfig.embedding_model)
            .join(IndexConfig, Deployment.index_config_id == IndexConfig.id)
            .join(Project, IndexConfig.project_id == Project.id)
            .where(
                Project.name.in_(project_names),
                Deployment.status == DeploymentStatus.ACTIVE.value,
            )
        )
        for row in result:
            dep, emb_provider, emb_model = row
            pname = dep.project_name
            cost_row = summary_by_project.get(pname, {})
            plan_stats.append(
                PlanStats(
                    plan_name=pname,
                    total_queries=cost_row.get("total_queries") or 0,
                    cache_hits=cost_row.get("cache_hits") or 0,
                    estimated_cost_usd=float(cost_row.get("estimated_cost_usd") or 0.0),
                    cache_enabled=dep.cache_enabled,
                    top_k=dep.top_k,
                    embedding_provider=emb_provider,
                    embedding_model=emb_model,
                )
            )

    recs = generate_recommendations(plan_stats)

    # Update Prometheus gauge
    metrics.recommendations_active.labels(category="all", priority="all").set(len(recs))

    return RecommendationsResponse(
        total=len(recs),
        items=[
            Recommendation(
                project_name=r.plan_name,
                category=r.category,
                priority=r.priority,
                message=r.message,
                potential_savings_pct=r.potential_savings_pct,
            )
            for r in recs
        ],
    )
