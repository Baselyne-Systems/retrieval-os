"""Pydantic schemas for the Webhooks domain."""

from __future__ import annotations

from datetime import datetime

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator

from retrieval_os.webhooks.events import WebhookEvent


class CreateWebhookSubscriptionRequest(BaseModel):
    url: AnyHttpUrl = Field(..., description="The HTTPS endpoint to deliver events to.")
    events: list[WebhookEvent] = Field(
        default_factory=list,
        description="Event types to subscribe to. Empty list subscribes to all events.",
    )
    secret: str | None = Field(
        default=None,
        description=(
            "Optional shared secret for HMAC-SHA256 request signing. "
            "Sent as X-Retrieval-OS-Signature: sha256=<hex> header."
        ),
    )
    description: str | None = Field(default=None, max_length=1024)

    @field_validator("events", mode="before")
    @classmethod
    def _deduplicate_events(cls, v: list) -> list:
        seen: set = set()
        return [e for e in v if not (e in seen or seen.add(e))]  # type: ignore[func-returns-value]


class WebhookSubscriptionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    url: str
    events: list[str]
    description: str | None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    # Secret is intentionally excluded from responses.


class WebhookSubscriptionListResponse(BaseModel):
    items: list[WebhookSubscriptionResponse]
    total: int
