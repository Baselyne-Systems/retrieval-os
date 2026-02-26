"""Pydantic schemas for the Cost Intelligence domain."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelPricingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    provider: str
    model: str
    cost_per_1k_tokens: float
    valid_from: datetime
    valid_until: datetime | None
    created_at: datetime


class AddModelPricingRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=255)
    cost_per_1k_tokens: float = Field(..., ge=0.0, description="USD per 1,000 tokens")


class CostEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_name: str
    index_config_version: int
    window_start: datetime
    window_end: datetime
    provider: str
    model: str
    total_queries: int
    cache_hits: int
    token_count: int
    estimated_cost_usd: float
    created_at: datetime
    updated_at: datetime


class CostSummaryRow(BaseModel):
    project_name: str
    total_queries: int
    cache_hits: int
    cache_hit_rate: float
    token_count: int
    estimated_cost_usd: float
    cost_per_query_usd: float


class CostSummaryResponse(BaseModel):
    since: datetime
    until: datetime
    rows: list[CostSummaryRow]
    grand_total_usd: float


class Recommendation(BaseModel):
    project_name: str
    category: str  # "cache", "model", "top_k"
    priority: str  # "high", "medium", "low"
    message: str
    potential_savings_pct: float | None = None


class RecommendationsResponse(BaseModel):
    total: int
    items: list[Recommendation]
