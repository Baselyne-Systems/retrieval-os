"""Unit tests for the serving path (no live Redis / Qdrant / models needed)."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest

from retrieval_os.serving.cache import _cache_key, cache_get, cache_set
from retrieval_os.serving.executor import RetrievedChunk
from retrieval_os.serving.schemas import ChunkResponse, QueryRequest, QueryResponse

# ── Cache key ─────────────────────────────────────────────────────────────────

class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = _cache_key("docs", 1, "what is RAG?", 10)
        k2 = _cache_key("docs", 1, "what is RAG?", 10)
        assert k1 == k2

    def test_different_query_different_key(self) -> None:
        k1 = _cache_key("docs", 1, "query A", 10)
        k2 = _cache_key("docs", 1, "query B", 10)
        assert k1 != k2

    def test_different_version_different_key(self) -> None:
        k1 = _cache_key("docs", 1, "same query", 10)
        k2 = _cache_key("docs", 2, "same query", 10)
        assert k1 != k2

    def test_different_plan_different_key(self) -> None:
        k1 = _cache_key("plan_a", 1, "same query", 10)
        k2 = _cache_key("plan_b", 1, "same query", 10)
        assert k1 != k2

    def test_different_top_k_different_key(self) -> None:
        k1 = _cache_key("docs", 1, "same query", 5)
        k2 = _cache_key("docs", 1, "same query", 10)
        assert k1 != k2

    def test_key_has_prefix(self) -> None:
        k = _cache_key("docs", 1, "query", 10)
        assert k.startswith("ros:qcache:")

    def test_key_uses_sha256(self) -> None:
        raw = "docs|1|query text|10"
        expected_digest = hashlib.sha256(raw.encode()).hexdigest()
        k = _cache_key("docs", 1, "query text", 10)
        assert k == f"ros:qcache:{expected_digest}"


# ── Cache get/set ─────────────────────────────────────────────────────────────

class TestCacheGetSet:
    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        return redis

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, mock_redis: AsyncMock) -> None:
        with patch("retrieval_os.serving.cache.get_redis", AsyncMock(return_value=mock_redis)):
            result = await cache_get("docs", 1, "query", 10)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_chunks(self, mock_redis: AsyncMock) -> None:
        chunks = [{"id": "1", "score": 0.9, "text": "hello", "metadata": {}}]
        mock_redis.get = AsyncMock(return_value=json.dumps(chunks))
        with patch("retrieval_os.serving.cache.get_redis", AsyncMock(return_value=mock_redis)):
            result = await cache_get("docs", 1, "query", 10)
        assert result == chunks

    @pytest.mark.asyncio
    async def test_cache_set_with_positive_ttl(self, mock_redis: AsyncMock) -> None:
        chunks = [{"id": "1", "score": 0.9, "text": "hello", "metadata": {}}]
        with patch("retrieval_os.serving.cache.get_redis", AsyncMock(return_value=mock_redis)):
            await cache_set("docs", 1, "query", 10, chunks, ttl_seconds=3600)
        mock_redis.set.assert_awaited_once()
        _, kwargs = mock_redis.set.call_args
        assert kwargs.get("ex") == 3600

    @pytest.mark.asyncio
    async def test_cache_set_skipped_when_ttl_zero(self, mock_redis: AsyncMock) -> None:
        with patch("retrieval_os.serving.cache.get_redis", AsyncMock(return_value=mock_redis)):
            await cache_set("docs", 1, "query", 10, [], ttl_seconds=0)
        mock_redis.set.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_get_returns_none_on_redis_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
        with patch("retrieval_os.serving.cache.get_redis", AsyncMock(return_value=mock_redis)):
            result = await cache_get("docs", 1, "query", 10)
        assert result is None



# ── RetrievedChunk ─────────────────────────────────────────────────────────────

class TestRetrievedChunk:
    def test_to_dict_has_all_fields(self) -> None:
        chunk = RetrievedChunk(
            id="abc", score=0.85, text="Some text", metadata={"source": "doc1"}
        )
        d = chunk.to_dict()
        assert d == {
            "id": "abc",
            "score": 0.85,
            "text": "Some text",
            "metadata": {"source": "doc1"},
        }


# ── Schemas ───────────────────────────────────────────────────────────────────

class TestQuerySchemas:
    def test_query_request_valid(self) -> None:
        r = QueryRequest(query="hello world")
        assert r.query == "hello world"
        assert r.metadata_filters is None

    def test_query_request_with_filters(self) -> None:
        r = QueryRequest(query="hello", metadata_filters={"lang": "en"})
        assert r.metadata_filters == {"lang": "en"}

    def test_query_request_empty_string_invalid(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            QueryRequest(query="")

    def test_query_response_fields(self) -> None:
        resp = QueryResponse(
            plan_name="docs",
            version=2,
            cache_hit=False,
            results=[
                ChunkResponse(id="1", score=0.9, text="hi", metadata={})
            ],
            result_count=1,
        )
        assert resp.result_count == 1
        assert resp.results[0].id == "1"
