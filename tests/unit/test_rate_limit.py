"""Unit tests for the Redis sliding-window rate limiter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.api.middleware.rate_limit import _WINDOW_SECONDS, check_rate_limit

# ── Fake Redis helpers ────────────────────────────────────────────────────────


def _make_fake_redis(zcard_result: int) -> MagicMock:
    """Return a mock Redis client whose pipeline reports *zcard_result* items."""
    pipe = AsyncMock()
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    # pipeline.execute() returns [zremrangebyscore, zadd, zcard, expire]
    pipe.execute = AsyncMock(return_value=[None, None, zcard_result, None])

    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    return redis


# ── check_rate_limit ──────────────────────────────────────────────────────────


class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_allowed_when_below_limit(self) -> None:
        fake_redis = _make_fake_redis(zcard_result=30)
        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=AsyncMock(return_value=fake_redis),
        ):
            allowed = await check_rate_limit("tenant-1", max_rpm=60)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_allowed_at_exact_limit(self) -> None:
        fake_redis = _make_fake_redis(zcard_result=60)
        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=AsyncMock(return_value=fake_redis),
        ):
            allowed = await check_rate_limit("tenant-1", max_rpm=60)
        assert allowed is True  # exactly at limit is allowed

    @pytest.mark.asyncio
    async def test_denied_when_over_limit(self) -> None:
        fake_redis = _make_fake_redis(zcard_result=61)
        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=AsyncMock(return_value=fake_redis),
        ):
            allowed = await check_rate_limit("tenant-1", max_rpm=60)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_allowed_for_zero_requests(self) -> None:
        fake_redis = _make_fake_redis(zcard_result=0)
        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=AsyncMock(return_value=fake_redis),
        ):
            allowed = await check_rate_limit("tenant-1", max_rpm=1)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_pipeline_commands_issued(self) -> None:
        fake_redis = _make_fake_redis(zcard_result=5)
        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=AsyncMock(return_value=fake_redis),
        ):
            await check_rate_limit("tenant-x", max_rpm=100)

        pipe = fake_redis.pipeline.return_value
        pipe.zremrangebyscore.assert_called_once()
        pipe.zadd.assert_called_once()
        pipe.zcard.assert_called_once()
        pipe.expire.assert_called_once()
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_key_includes_tenant_id(self) -> None:
        fake_redis = _make_fake_redis(zcard_result=0)
        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=AsyncMock(return_value=fake_redis),
        ):
            await check_rate_limit("my-tenant", max_rpm=10)

        pipe = fake_redis.pipeline.return_value
        # The first arg to zremrangebyscore should be the Redis key
        key_used = pipe.zremrangebyscore.call_args[0][0]
        assert "my-tenant" in key_used

    @pytest.mark.asyncio
    async def test_different_tenants_use_different_keys(self) -> None:
        calls: list[str] = []

        async def mock_get_redis():  # noqa: ANN201
            fake = _make_fake_redis(zcard_result=5)
            original_pipe = fake.pipeline.return_value

            def capture_key(key, *args, **kwargs):  # noqa: ANN001, ANN201
                calls.append(key)
                return original_pipe

            original_pipe.zremrangebyscore = capture_key
            return fake

        with patch(
            "retrieval_os.api.middleware.rate_limit.get_redis",
            new=mock_get_redis,
        ):
            await check_rate_limit("tenant-a", max_rpm=10)
            await check_rate_limit("tenant-b", max_rpm=10)

        assert len(calls) == 2
        assert calls[0] != calls[1]


# ── Window constant ───────────────────────────────────────────────────────────


class TestWindowConstant:
    def test_window_is_60_seconds(self) -> None:
        assert _WINDOW_SECONDS == 60


# ── Middleware integration ────────────────────────────────────────────────────


class TestRateLimitMiddlewareSkips:
    @pytest.mark.asyncio
    async def test_skips_when_disabled(self) -> None:
        """When rate_limit_enabled=False the middleware passes through."""
        from unittest.mock import patch as _patch

        # Build a minimal ASGI app with only RateLimitMiddleware
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from retrieval_os.api.middleware.rate_limit import RateLimitMiddleware

        async def homepage(request: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(RateLimitMiddleware)

        with _patch("retrieval_os.api.middleware.rate_limit.settings") as mock_settings:
            mock_settings.rate_limit_enabled = False
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/")
            assert resp.status_code == 200
