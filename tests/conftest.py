"""Shared pytest fixtures for retrieval-os test suite."""

import pytest
from httpx import ASGITransport, AsyncClient

from retrieval_os.api.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Async test client against the FastAPI app (no live infrastructure)."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
