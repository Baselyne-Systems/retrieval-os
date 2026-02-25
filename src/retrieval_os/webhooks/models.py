"""ORM model for WebhookSubscription."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from retrieval_os.core.database import Base


class WebhookSubscription(Base):
    """A registered endpoint that receives outbound event notifications.

    ``events`` is a JSON list of :class:`~retrieval_os.webhooks.events.WebhookEvent`
    values.  An empty list means "subscribe to all events".
    """

    __tablename__ = "webhook_subscriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    # JSON list of event strings; [] = subscribe to all events
    events: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    # Optional HMAC-SHA256 secret for payload signing (stored as plaintext —
    # treat the same as an API key; rotate regularly).
    secret: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
