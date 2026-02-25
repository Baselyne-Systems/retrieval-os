"""Unit tests for the Webhooks domain (models, schemas, delivery helpers)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from retrieval_os.webhooks.delivery import _deliver, _sign_payload, fire_webhook_event
from retrieval_os.webhooks.events import WebhookEvent
from retrieval_os.webhooks.models import WebhookSubscription
from retrieval_os.webhooks.schemas import (
    CreateWebhookSubscriptionRequest,
    WebhookSubscriptionResponse,
)

# ── WebhookEvent ──────────────────────────────────────────────────────────────


class TestWebhookEvent:
    def test_event_values(self) -> None:
        assert WebhookEvent.DEPLOYMENT_STATUS_CHANGED == "deployment.status_changed"
        assert WebhookEvent.EVAL_REGRESSION_DETECTED == "eval.regression_detected"
        assert WebhookEvent.PLAN_VERSION_CREATED == "plan.version_created"
        assert WebhookEvent.COST_THRESHOLD_EXCEEDED == "cost.threshold_exceeded"

    def test_is_str(self) -> None:
        # StrEnum values are plain strings
        assert isinstance(WebhookEvent.DEPLOYMENT_STATUS_CHANGED, str)

    def test_all_four_events_defined(self) -> None:
        assert len(WebhookEvent) == 4


# ── WebhookSubscription ORM ───────────────────────────────────────────────────


class TestWebhookSubscriptionModel:
    def test_constructor_all_events(self) -> None:
        now = datetime.now(UTC)
        sub = WebhookSubscription(
            id="ws-001",
            url="https://example.com/hook",
            events=[],
            secret=None,
            description="all events",
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        assert sub.url == "https://example.com/hook"
        assert sub.events == []
        assert sub.is_active is True

    def test_constructor_specific_events(self) -> None:
        now = datetime.now(UTC)
        sub = WebhookSubscription(
            id="ws-002",
            url="https://example.com/hook",
            events=["deployment.status_changed"],
            secret="mysecret",
            description=None,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        assert "deployment.status_changed" in sub.events
        assert sub.secret == "mysecret"

    def test_inactive_subscription(self) -> None:
        now = datetime.now(UTC)
        sub = WebhookSubscription(
            id="ws-003",
            url="https://example.com/hook",
            events=[],
            secret=None,
            description=None,
            is_active=False,
            created_at=now,
            updated_at=now,
        )
        assert sub.is_active is False


# ── Schemas ───────────────────────────────────────────────────────────────────


class TestCreateWebhookSubscriptionRequest:
    def test_valid_url_no_events(self) -> None:
        req = CreateWebhookSubscriptionRequest(url="https://example.com/hook")
        assert "example.com" in str(req.url)
        assert "hook" in str(req.url)
        assert req.events == []
        assert req.secret is None

    def test_valid_with_events(self) -> None:
        req = CreateWebhookSubscriptionRequest(
            url="https://hooks.example.com/cb",
            events=[WebhookEvent.DEPLOYMENT_STATUS_CHANGED],
        )
        assert WebhookEvent.DEPLOYMENT_STATUS_CHANGED in req.events

    def test_deduplicates_events(self) -> None:
        req = CreateWebhookSubscriptionRequest(
            url="https://hooks.example.com/cb",
            events=[
                WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
                WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
            ],
        )
        assert len(req.events) == 1

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(Exception):
            CreateWebhookSubscriptionRequest(url="not-a-url")

    def test_with_secret(self) -> None:
        req = CreateWebhookSubscriptionRequest(
            url="https://example.com/hook",
            secret="supersecret",
        )
        assert req.secret == "supersecret"

    def test_description_stored(self) -> None:
        req = CreateWebhookSubscriptionRequest(
            url="https://example.com/hook",
            description="My webhook",
        )
        assert req.description == "My webhook"


class TestWebhookSubscriptionResponse:
    def test_no_secret_field(self) -> None:
        """Secret must not appear in the response schema."""
        now = datetime.now(UTC)
        resp = WebhookSubscriptionResponse(
            id="ws-001",
            url="https://example.com/hook",
            events=[],
            description=None,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        assert not hasattr(resp, "secret")
        data = resp.model_dump()
        assert "secret" not in data

    def test_from_orm_model(self) -> None:
        now = datetime.now(UTC)
        sub = WebhookSubscription(
            id="ws-001",
            url="https://example.com/hook",
            events=["deployment.status_changed"],
            secret="hidden",
            description="desc",
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        resp = WebhookSubscriptionResponse.model_validate(sub)
        assert resp.id == "ws-001"
        assert resp.events == ["deployment.status_changed"]
        assert not hasattr(resp, "secret")


# ── _sign_payload ─────────────────────────────────────────────────────────────


class TestSignPayload:
    def test_produces_sha256_prefix(self) -> None:
        sig = _sign_payload("mysecret", b"hello")
        assert sig.startswith("sha256=")

    def test_correct_hmac(self) -> None:
        secret = "topsecret"
        payload = b'{"event": "test"}'
        expected_hex = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert _sign_payload(secret, payload) == f"sha256={expected_hex}"

    def test_deterministic(self) -> None:
        sig1 = _sign_payload("s", b"data")
        sig2 = _sign_payload("s", b"data")
        assert sig1 == sig2

    def test_different_secrets_differ(self) -> None:
        assert _sign_payload("secret1", b"data") != _sign_payload("secret2", b"data")

    def test_different_payloads_differ(self) -> None:
        assert _sign_payload("secret", b"payload1") != _sign_payload("secret", b"payload2")


# ── _deliver ──────────────────────────────────────────────────────────────────


class TestDeliver:
    @pytest.mark.asyncio
    async def test_posts_to_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await _deliver("https://example.com/hook", None, {"event": "test"})

        mock_client.post.assert_awaited_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[0][0] == "https://example.com/hook"

    @pytest.mark.asyncio
    async def test_adds_signature_when_secret_given(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await _deliver("https://example.com/hook", "mysecret", {"event": "test"})

        headers_sent = mock_client.post.call_args[1]["headers"]
        assert "X-Retrieval-OS-Signature" in headers_sent
        assert headers_sent["X-Retrieval-OS-Signature"].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_no_signature_when_no_secret(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 204

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await _deliver("https://example.com/hook", None, {"event": "test"})

        headers_sent = mock_client.post.call_args[1]["headers"]
        assert "X-Retrieval-OS-Signature" not in headers_sent

    @pytest.mark.asyncio
    async def test_retries_on_5xx(self) -> None:
        mock_5xx = MagicMock()
        mock_5xx.status_code = 500
        mock_ok = MagicMock()
        mock_ok.status_code = 200

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[mock_5xx, mock_ok])
            mock_client_cls.return_value = mock_client

            with patch("retrieval_os.webhooks.delivery.asyncio.sleep", new=AsyncMock()):
                await _deliver("https://example.com/hook", None, {}, max_attempts=2)

        assert mock_client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self) -> None:
        mock_5xx = MagicMock()
        mock_5xx.status_code = 503

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_5xx)
            mock_client_cls.return_value = mock_client

            with patch("retrieval_os.webhooks.delivery.asyncio.sleep", new=AsyncMock()):
                # Should not raise despite all 5xx responses
                await _deliver("https://example.com/hook", None, {}, max_attempts=2)

        assert mock_client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_handles_network_error_gracefully(self) -> None:
        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client

            with patch("retrieval_os.webhooks.delivery.asyncio.sleep", new=AsyncMock()):
                # Must not raise
                await _deliver("https://example.com/hook", None, {}, max_attempts=2)

    @pytest.mark.asyncio
    async def test_does_not_retry_on_4xx(self) -> None:
        mock_4xx = MagicMock()
        mock_4xx.status_code = 404

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_4xx)
            mock_client_cls.return_value = mock_client

            await _deliver("https://example.com/hook", None, {}, max_attempts=3)

        # 4xx is treated as "not 5xx" so it exits after first attempt
        assert mock_client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_payload_is_valid_json(self) -> None:
        captured: list[bytes] = []
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        async def capture_post(url, *, content, headers):  # noqa: ANN001
            captured.append(content)
            return mock_resp

        with patch("retrieval_os.webhooks.delivery.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = capture_post
            mock_client_cls.return_value = mock_client

            await _deliver("https://example.com/hook", None, {"key": "value"})

        assert len(captured) == 1
        parsed = json.loads(captured[0])
        assert parsed["key"] == "value"


# ── fire_webhook_event ────────────────────────────────────────────────────────


class TestFireWebhookEvent:
    @pytest.mark.asyncio
    async def test_creates_tasks_for_each_subscription(self) -> None:
        now = datetime.now(UTC)
        subs = [
            WebhookSubscription(
                id=f"ws-00{i}",
                url=f"https://example.com/hook{i}",
                events=[],
                secret=None,
                description=None,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            for i in range(3)
        ]

        mock_session = MagicMock()
        tasks_created: list = []

        with (
            patch(
                "retrieval_os.webhooks.delivery.webhook_repo.get_subscriptions_for_event",
                new=AsyncMock(return_value=subs),
            ),
            patch(
                "retrieval_os.webhooks.delivery.asyncio.create_task",
                side_effect=lambda coro, **kw: tasks_created.append(coro) or MagicMock(),
            ),
        ):
            await fire_webhook_event(
                WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
                {"status": "active"},
                mock_session,
            )

        assert len(tasks_created) == 3
        # Clean up unawaited coroutines to avoid warnings
        for coro in tasks_created:
            coro.close()

    @pytest.mark.asyncio
    async def test_no_tasks_when_no_subscriptions(self) -> None:
        mock_session = MagicMock()
        tasks_created: list = []

        with (
            patch(
                "retrieval_os.webhooks.delivery.webhook_repo.get_subscriptions_for_event",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "retrieval_os.webhooks.delivery.asyncio.create_task",
                side_effect=lambda coro, **kw: tasks_created.append(coro) or MagicMock(),
            ),
        ):
            await fire_webhook_event(
                WebhookEvent.EVAL_REGRESSION_DETECTED,
                {},
                mock_session,
            )

        assert len(tasks_created) == 0

    @pytest.mark.asyncio
    async def test_payload_includes_event_and_timestamp(self) -> None:
        now = datetime.now(UTC)
        sub = WebhookSubscription(
            id="ws-001",
            url="https://example.com/hook",
            events=[],
            secret=None,
            description=None,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

        payloads_sent: list[dict] = []
        mock_session = MagicMock()

        async def fake_deliver(url, secret, payload, **kw):  # noqa: ANN001
            payloads_sent.append(payload)

        with (
            patch(
                "retrieval_os.webhooks.delivery.webhook_repo.get_subscriptions_for_event",
                new=AsyncMock(return_value=[sub]),
            ),
            patch(
                "retrieval_os.webhooks.delivery.asyncio.create_task",
                side_effect=lambda coro, **kw: asyncio.ensure_future(coro),
            ),
            patch("retrieval_os.webhooks.delivery._deliver", new=fake_deliver),
        ):
            await fire_webhook_event(
                WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
                {"plan_name": "acme"},
                mock_session,
            )
            # Let the event loop run the tasks
            await asyncio.sleep(0)

        assert len(payloads_sent) == 1
        assert payloads_sent[0]["event"] == "deployment.status_changed"
        assert "timestamp" in payloads_sent[0]
        assert payloads_sent[0]["data"]["plan_name"] == "acme"
