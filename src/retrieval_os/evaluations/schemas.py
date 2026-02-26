"""Pydantic schemas for the Evaluation Engine domain."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QueueEvalJobRequest(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=255)
    index_config_version: int = Field(..., ge=1)
    dataset_uri: str = Field(..., min_length=1, description="S3 URI of the JSONL eval dataset")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to retrieve per query")
    created_by: str = Field(..., min_length=1, max_length=255)


class EvalJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_name: str
    index_config_version: int
    status: str
    dataset_uri: str
    top_k: int

    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    context_quality_score: float | None

    total_queries: int | None
    failed_queries: int | None
    error_message: str | None

    regression_detected: bool | None
    regression_detail: list | None

    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    created_by: str


class EvalJobListResponse(BaseModel):
    items: list[EvalJobResponse]
    total: int
