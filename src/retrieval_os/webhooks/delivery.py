"""Outbound webhook delivery with HMAC-SHA256 signing and retry.

Delivery algorithm:
    1. Serialise the event payload to JSON bytes.
    2. If the subscription has a secret, compute
       ``HMAC-SHA256(secret, payload)`` and attach it as the
       ``X-Retrieval-OS-Signature: sha256=<hex>`` header.
    3. POST to the subscriber URL (5-second connect + 10-second read timeout).
    4. Retry up to ``MAX_ATTEMPTS`` times on network errors or 5xx responses,
       with exponential back-off (1 s, 2 s, 4 s …).
    5. Log final success or failure; never raise to the caller.

``fire_webhook_event`` is the public entry point.  It queries the database for
matching subscriptions within the *current* request session, then schedules
each delivery as a fire-and-forget ``asyncio.Task`` so the HTTP call happens
outside the request/response cycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.webhooks.events import WebhookEvent
from retrieval_os.webhooks.repository import webhook_repo

log = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 5.0   # seconds
_READ_TIMEOUT = 10.0     # seconds
_MAX_ATTEMPTS = 3


def _sign_payload(secret: str, payload_bytes: bytes) -> str:
    """Return ``sha256=<hex>`` HMAC signature."""
    digest = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


async def _deliver(
    url: str,
    secret: str | None,
    payload: dict,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
) -> None:
    """POST *payload* to *url* with optional HMAC signing.  Retries on 5xx."""
    payload_bytes = json.dumps(payload, default=str).encode()
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "Retrieval-OS-Webhooks/1.0",
    }
    if secret:
        headers["X-Retrieval-OS-Signature"] = _sign_payload(secret, payload_bytes)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=10.0, pool=5.0)
    ) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.post(url, content=payload_bytes, headers=headers)
                if resp.status_code < 500:
                    log.info(
                        "webhook.delivered",
                        extra={
                            "url": url,
                            "status": resp.status_code,
                            "attempt": attempt,
                        },
                    )
                    return
                log.warning(
                    "webhook.delivery_5xx",
                    extra={
                        "url": url,
                        "status": resp.status_code,
                        "attempt": attempt,
                    },
                )
            except Exception as exc:
                log.warning(
                    "webhook.delivery_error",
                    extra={"url": url, "attempt": attempt, "error": str(exc)},
                )

            if attempt < max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))

    log.error(
        "webhook.delivery_failed",
        extra={"url": url, "attempts": max_attempts},
    )


async def fire_webhook_event(
    event_type: WebhookEvent,
    data: dict,
    session: AsyncSession,
) -> None:
    """Query matching subscriptions and schedule fire-and-forget deliveries.

    The *session* must still be open when this is called so we can load
    subscriptions.  Each delivery task then runs independently of the session.
    """
    subscriptions = await webhook_repo.get_subscriptions_for_event(
        session, str(event_type)
    )
    if not subscriptions:
        return

    payload = {
        "event": str(event_type),
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data,
    }

    for sub in subscriptions:
        asyncio.create_task(
            _deliver(str(sub.url), sub.secret, payload),
            name=f"webhook.{event_type}.{sub.id}",
        )
        log.debug(
            "webhook.scheduled",
            extra={"event": str(event_type), "subscription_id": sub.id, "url": sub.url},
        )
