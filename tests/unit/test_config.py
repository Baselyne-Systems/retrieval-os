"""Tests for settings loading."""

import pytest

from retrieval_os.core.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.app_name == "retrieval-os"
    assert s.environment == "development"
    assert s.database_pool_size == 10


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_POOL_SIZE", "25")
    s = Settings()
    assert s.environment == "production"
    assert s.database_pool_size == 25


def test_database_url_contains_asyncpg() -> None:
    s = Settings()
    assert "asyncpg" in s.database_url


def test_redis_url_scheme() -> None:
    s = Settings()
    assert s.redis_url.startswith("redis://")
