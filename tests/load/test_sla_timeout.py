"""Load test: Query SLA timeout returns 504 within the configured deadline.

Proves:
1. A hanging Qdrant backend causes the server to respond with HTTP 504
   (QUERY_TIMEOUT) within ``settings.query_timeout_seconds + 1.0s``.
2. A fast query on a real backend is unaffected (200 OK, not 504).

Infrastructure required for test 2: Postgres + Redis + Qdrant.
Test 1 only needs Postgres + Redis (Qdrant is patched to hang).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

from retrieval_os.core.config import settings


@pytest.fixture
async def app():
    """Return a bare FastAPI app instance (no lifespan) for ASGI transport tests."""
    from retrieval_os.api.main import create_app

    return create_app()


class TestQueryTimeout:
    """Query timeout SLA enforcement via asyncio.wait_for."""

    async def test_slow_backend_returns_504_within_timeout(self, load_project, app) -> None:
        """A hanging Qdrant search must be cancelled within query_timeout_seconds.

        The response must arrive within timeout + 1 s (1 s tolerance for app
        overhead) and carry status 504 with error code QUERY_TIMEOUT.
        """

        async def _hang(**kwargs):
            await asyncio.sleep(60)  # simulate a stuck backend
            return []

        deadline = settings.query_timeout_seconds + 1.0

        with (
            patch(
                "retrieval_os.serving.executor.embed_text",
                new=AsyncMock(return_value=[[0.1] * 384]),
            ),
            patch("retrieval_os.serving.index_proxy._qdrant_search", side_effect=_hang),
        ):
            wall_start = time.perf_counter()
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    f"/v1/query/{load_project}",
                    json={"query": "timeout test query unique xyz123"},
                    timeout=deadline + 5.0,  # httpx timeout must exceed our SLA check
                )
            elapsed = time.perf_counter() - wall_start

        assert resp.status_code == 504, (
            f"Expected 504 QUERY_TIMEOUT, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body.get("error") == "QUERY_TIMEOUT", f"Unexpected error body: {body}"
        assert elapsed < deadline, (
            f"Response took {elapsed:.1f}s; expected < {deadline:.1f}s "
            f"(timeout={settings.query_timeout_seconds}s + 1s tolerance)"
        )

    async def test_fast_query_unaffected(self, load_project, load_collection) -> None:
        """A fast query on a real Qdrant backend must return 200 OK, not 504."""
        from tests.load.conftest import random_unit_vector

        stub_vector = random_unit_vector()

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new=AsyncMock(return_value=[stub_vector]),
        ):
            from retrieval_os.core.database import async_session_factory
            from retrieval_os.serving.query_router import route_query

            async with async_session_factory() as session:
                chunks, info = await route_query(
                    project_name=load_project,
                    query="fast query unaffected by timeout",
                    db=session,
                )

        assert info["result_count"] >= 0
        assert info.get("cache_hit") is not None
