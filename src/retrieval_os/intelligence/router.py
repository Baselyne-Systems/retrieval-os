"""FastAPI router for the Cost Intelligence domain."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.intelligence import service
from retrieval_os.intelligence.schemas import (
    AddModelPricingRequest,
    CostSummaryResponse,
    ModelPricingResponse,
    RecommendationsResponse,
)

router = APIRouter(prefix="/v1/intelligence", tags=["intelligence"])


@router.get("/model-pricing", response_model=list[ModelPricingResponse])
async def list_model_pricing(
    db: AsyncSession = Depends(get_db),
) -> list[ModelPricingResponse]:
    """List all current (non-expired) model pricing entries."""
    return await service.list_model_pricing(db)


@router.post("/model-pricing", status_code=201, response_model=ModelPricingResponse)
async def add_model_pricing(
    request: AddModelPricingRequest,
    db: AsyncSession = Depends(get_db),
) -> ModelPricingResponse:
    """Add or update the price for a provider/model pair.

    If a current pricing entry already exists for this provider+model, it is
    expired and replaced by the new entry.
    """
    return await service.add_model_pricing(db, request)


@router.get("/cost/summary", response_model=CostSummaryResponse)
async def get_cost_summary(
    plan_name: str | None = Query(None, description="Filter to a single plan"),
    since: datetime | None = Query(None, description="Start of window (default: 7 days ago)"),
    until: datetime | None = Query(None, description="End of window (default: now)"),
    db: AsyncSession = Depends(get_db),
) -> CostSummaryResponse:
    """Return aggregated cost summary per plan for the requested time window."""
    return await service.get_cost_summary(db, since=since, until=until, plan_name=plan_name)


@router.get("/cost/entries", response_model=dict)
async def list_cost_entries(
    plan_name: str | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List raw hourly cost entries, newest first."""
    entries, total = await service.list_cost_entries(
        db, plan_name=plan_name, since=since, until=until, limit=limit, offset=offset
    )
    return {"items": [e.model_dump() for e in entries], "total": total}


@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(
    plan_name: str | None = Query(None, description="Scope to a single plan"),
    db: AsyncSession = Depends(get_db),
) -> RecommendationsResponse:
    """Return rule-based cost and performance recommendations.

    Rules checked:
    - Cache disabled → enable semantic cache
    - Cache hit rate < 30 % → tune TTL or review query diversity
    - Cost per query > $0.001 → consider a smaller embedding model
    - top_k > 20 → consider reducing retrieval width
    """
    return await service.get_recommendations(db, plan_name=plan_name)
