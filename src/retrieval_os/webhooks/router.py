"""REST endpoints for the Webhooks domain.

Routes
------
POST   /v1/webhooks              — register a new subscription
GET    /v1/webhooks              — list all subscriptions
GET    /v1/webhooks/{id}         — fetch a single subscription
DELETE /v1/webhooks/{id}         — delete a subscription
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.core.ids import uuid7
from retrieval_os.webhooks.models import WebhookSubscription
from retrieval_os.webhooks.repository import webhook_repo
from retrieval_os.webhooks.schemas import (
    CreateWebhookSubscriptionRequest,
    WebhookSubscriptionListResponse,
    WebhookSubscriptionResponse,
)

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])


@router.post("", response_model=WebhookSubscriptionResponse, status_code=201)
async def create_webhook(
    request: CreateWebhookSubscriptionRequest,
    session: AsyncSession = Depends(get_db),
) -> WebhookSubscriptionResponse:
    now = datetime.now(UTC)
    sub = WebhookSubscription(
        id=str(uuid7()),
        url=str(request.url),
        events=[str(e) for e in request.events],
        secret=request.secret,
        description=request.description,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    sub = await webhook_repo.create(session, sub)
    return WebhookSubscriptionResponse.model_validate(sub)


@router.get("", response_model=WebhookSubscriptionListResponse)
async def list_webhooks(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> WebhookSubscriptionListResponse:
    items, total = await webhook_repo.list_all(session, offset=offset, limit=limit)
    return WebhookSubscriptionListResponse(
        items=[WebhookSubscriptionResponse.model_validate(i) for i in items],
        total=total,
    )


@router.get("/{webhook_id}", response_model=WebhookSubscriptionResponse)
async def get_webhook(
    webhook_id: str,
    session: AsyncSession = Depends(get_db),
) -> WebhookSubscriptionResponse:
    sub = await webhook_repo.get(session, webhook_id)
    if sub is None:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")
    return WebhookSubscriptionResponse.model_validate(sub)


@router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: str,
    session: AsyncSession = Depends(get_db),
) -> None:
    deleted = await webhook_repo.delete(session, webhook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found")
