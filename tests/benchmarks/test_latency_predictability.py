"""Benchmark: Latency Predictability

Proves the in-process overhead of the serving layer — the code that runs on
every query before and after the actual embedding and vector-search calls —
is small enough to not dominate query latency at scale.

What we measure
---------------
The two in-process steps that run on every query:

1. Serving config JSON round-trip — the query router reads the serving config
   from Redis as a JSON string and deserialises it into a dict. At 10 000 QPS,
   this must not add more than a few milliseconds of total CPU overhead.

2. Redis key derivation — _project_redis_key is called once per query to
   build the lookup key. At scale the key format must be stable and derivation
   must be O(1) with a negligible constant.

3. Serving config merge (cache-miss path) — on a Redis miss, the router merges
   IndexConfig fields + Deployment fields into a single serving dict. This must
   be fast enough that even 100% cache-miss scenarios don't degrade throughput.

These are pure-Python, infrastructure-free measurements. Network latency (Redis
round-trip, embedding API, vector DB) is excluded — those are measured in load
tests against a live stack.

Scale targets
-------------
- JSON round-trip for 10 000 serving configs    in < 500 ms   → < 0.05 ms/query
- Redis key derivation for 100 000 queries      in < 100 ms   → < 0.001 ms/query
- Serving config merge for 100 000 operations   in < 500 ms   → < 0.005 ms/merge
"""

from __future__ import annotations

import json
import time

from retrieval_os.serving.query_router import _project_redis_key

# ── Shared fixtures ───────────────────────────────────────────────────────────

_SERVING_CONFIG = {
    "project_name": "docs",
    "index_config_version": 3,
    "embedding_provider": "sentence_transformers",
    "embedding_model": "BAAI/bge-m3",
    "embedding_normalize": True,
    "embedding_batch_size": 32,
    "index_backend": "qdrant",
    "index_collection": "docs_v3",
    "distance_metric": "cosine",
    "top_k": 10,
    "reranker": None,
    "rerank_top_k": None,
    "metadata_filters": None,
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,
    "hybrid_alpha": None,
}

_SERVING_CONFIG_JSON = json.dumps(_SERVING_CONFIG, default=str)


# ── JSON round-trip throughput ────────────────────────────────────────────────


class TestServingConfigJsonRoundTrip:
    def test_10k_json_deserialise_under_500ms(self) -> None:
        """Deserialising a serving config JSON string 10 000 times must take < 500 ms.

        The query router does this on every query where the config is in Redis.
        At 10 000 QPS this represents 10 000 JSON parses per second.
        """
        raw = _SERVING_CONFIG_JSON.encode()

        start = time.perf_counter()
        for _ in range(10_000):
            config = json.loads(raw)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"10 000 JSON deserialise ops took {elapsed:.3f}s; must be < 0.5s "
            f"({elapsed / 10_000 * 1_000:.4f} ms/op)"
        )
        assert config["project_name"] == "docs"

    def test_10k_json_serialise_under_500ms(self) -> None:
        """Serialising a serving config dict 10 000 times must take < 500 ms.

        The query router writes the config to Redis on every cache-miss path.
        """
        start = time.perf_counter()
        for _ in range(10_000):
            json.dumps(_SERVING_CONFIG, default=str)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"10 000 JSON serialise ops took {elapsed:.3f}s; must be < 0.5s "
            f"({elapsed / 10_000 * 1_000:.4f} ms/op)"
        )

    def test_json_roundtrip_is_lossless(self) -> None:
        """Serialise → deserialise must reproduce the original config exactly."""
        serialised = json.dumps(_SERVING_CONFIG, default=str)
        recovered = json.loads(serialised)
        for key, value in _SERVING_CONFIG.items():
            assert recovered[key] == value, (
                f"Field '{key}': original={value!r}, recovered={recovered[key]!r}"
            )


# ── Redis key derivation throughput ──────────────────────────────────────────


class TestRedisKeyDerivation:
    def test_100k_key_derivations_under_100ms(self) -> None:
        """Deriving 100 000 Redis keys must take < 100 ms.

        _project_redis_key is called once per query in the hot path. At 10 000 QPS
        over 10 seconds, 100 000 key derivations must not consume more than 100 ms
        of CPU time (< 1% overhead budget against a 10 ms average query latency).
        """
        project_names = [f"project-{i % 500}" for i in range(100_000)]

        start = time.perf_counter()
        for name in project_names:
            _project_redis_key(name)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, (
            f"100 000 Redis key derivations took {elapsed * 1000:.2f}ms; must be < 100ms "
            f"({elapsed / 100_000 * 1_000_000:.2f} µs/key)"
        )

    def test_key_format_is_stable(self) -> None:
        """The Redis key format must not change — existing cached configs depend on it."""
        assert _project_redis_key("my-docs") == "ros:project:my-docs:active"
        assert _project_redis_key("wiki-search") == "ros:project:wiki-search:active"
        assert _project_redis_key("prod") == "ros:project:prod:active"


# ── Serving config merge throughput ──────────────────────────────────────────


class TestServingConfigMerge:
    def _make_index_config(self, *, version: int = 1) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            embedding_provider="sentence_transformers",
            embedding_model=f"model-v{version}",
            embedding_normalize=True,
            embedding_batch_size=32,
            index_backend="qdrant",
            index_collection=f"col_v{version}",
            distance_metric="cosine",
        )

    def _make_deployment(self) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            index_config_version=1,
            top_k=10,
            reranker=None,
            rerank_top_k=None,
            metadata_filters=None,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            hybrid_alpha=None,
        )

    def test_100k_merges_under_500ms(self) -> None:
        """Building a serving config dict from IndexConfig + Deployment 100 000 times
        must take < 500 ms.

        This operation runs on the cache-miss path (every time Redis is cold or a new
        deployment is made). At high deployment frequency or after a Redis flush,
        many queries hit this path simultaneously.
        """
        from retrieval_os.deployments.service import _build_serving_config

        index_config = self._make_index_config()
        deployment = self._make_deployment()

        start = time.perf_counter()
        for _ in range(100_000):
            _build_serving_config("my-docs", deployment, index_config)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"100 000 serving config merges took {elapsed:.3f}s; must be < 0.5s "
            f"({elapsed / 100_000 * 1_000:.4f} ms/merge)"
        )

    def test_merged_config_has_all_16_fields(self) -> None:
        """The merged serving config must always carry all 16 fields.

        The executor reads every one of these fields on the hot path. A missing
        field causes a KeyError that kills the query.
        """
        from retrieval_os.deployments.service import _build_serving_config

        required = {
            "project_name",
            "index_config_version",
            "embedding_provider",
            "embedding_model",
            "embedding_normalize",
            "embedding_batch_size",
            "index_backend",
            "index_collection",
            "distance_metric",
            "top_k",
            "reranker",
            "rerank_top_k",
            "metadata_filters",
            "cache_enabled",
            "cache_ttl_seconds",
            "hybrid_alpha",
        }

        config = _build_serving_config(
            "my-docs", self._make_deployment(), self._make_index_config()
        )
        missing = required - set(config.keys())
        assert not missing, f"Serving config is missing fields: {missing}"
