"""Tests for health endpoints — mocks all infrastructure dependencies."""

from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient


@pytest.fixture
def mock_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch DB and Redis checks so tests don't need live infrastructure."""
    monkeypatch.setattr("retrieval_os.api.health.check_db_connection", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "retrieval_os.api.health.check_redis_connection", AsyncMock(return_value=True)
    )


async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_all_ok(client: AsyncClient, mock_deps: None) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["redis"] == "ok"


async def test_ready_degraded_when_db_down(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "retrieval_os.api.health.check_db_connection", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        "retrieval_os.api.health.check_redis_connection", AsyncMock(return_value=True)
    )
    response = await client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["postgres"] == "error"


async def test_metrics_endpoint_returns_prometheus_format(
    client: AsyncClient,
) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


async def test_info_endpoint(client: AsyncClient) -> None:
    response = await client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert "version" in body
    assert "environment" in body
    assert "service" in body


async def test_request_id_header_returned(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "x-request-id" in response.headers


async def test_custom_request_id_echoed(client: AsyncClient) -> None:
    custom_id = "test-id-123"
    response = await client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers.get("x-request-id") == custom_id
