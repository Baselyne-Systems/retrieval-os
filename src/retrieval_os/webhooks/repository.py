"""Database repository for WebhookSubscription."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.webhooks.models import WebhookSubscription


class WebhookRepository:
    async def create(
        self, session: AsyncSession, subscription: WebhookSubscription
    ) -> WebhookSubscription:
        session.add(subscription)
        await session.flush()
        await session.refresh(subscription)
        return subscription

    async def get(
        self, session: AsyncSession, subscription_id: str
    ) -> WebhookSubscription | None:
        return await session.get(WebhookSubscription, subscription_id)

    async def list_all(
        self,
        session: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[WebhookSubscription], int]:
        count_result = await session.execute(
            select(func.count()).select_from(WebhookSubscription)
        )
        total: int = count_result.scalar_one()

        result = await session.execute(
            select(WebhookSubscription)
            .order_by(WebhookSubscription.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def get_subscriptions_for_event(
        self, session: AsyncSession, event_type: str
    ) -> list[WebhookSubscription]:
        """Return active subscriptions whose events list is empty (all-events) OR
        contains the given *event_type*.

        Uses a JSON containment check via a Python-side filter after loading all
        active subscriptions.  Webhook subscription counts are expected to stay
        small (<<1000) so an in-memory filter is acceptable; add a DB-side JSON
        index for larger deployments.
        """
        result = await session.execute(
            select(WebhookSubscription).where(WebhookSubscription.is_active.is_(True))
        )
        all_active = list(result.scalars().all())
        return [
            sub
            for sub in all_active
            if not sub.events or event_type in sub.events
        ]

    async def delete(
        self, session: AsyncSession, subscription_id: str
    ) -> bool:
        sub = await self.get(session, subscription_id)
        if sub is None:
            return False
        await session.delete(sub)
        await session.flush()
        return True


webhook_repo = WebhookRepository()
