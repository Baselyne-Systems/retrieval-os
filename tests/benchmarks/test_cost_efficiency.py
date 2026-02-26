"""Benchmark: Cost Efficiency

Proves the cache layer can sustain the throughput required to absorb repeated
queries without the key-generation step becoming the bottleneck.

What we measure
---------------
1. Cache key generation throughput — at 10 000 QPS, the SHA-256 key derivation
   must add negligible overhead (< 0.1 ms per key).

2. Collision resistance at scale — 100 000 distinct (project, version, query,
   top_k) tuples must produce 100 000 distinct keys with zero collisions.
   A collision would cause one query's cached results to be silently served
   for a different query.

3. Key space isolation between projects — queries with the same text but
   different project names must produce different keys so tenant data never
   crosses project boundaries.

4. Key space isolation between index versions — queries with the same text but
   different index_config_version values must produce different keys so cached
   results from an old index are never served after a new deployment.

Scale targets
-------------
- 100 000 key generations   in < 2 s        → < 0.02 ms/key
- 100 000 distinct inputs   → 0 collisions
- Cross-project isolation   → verified across 1 000 project pairs
- Cross-version isolation   → verified across 1 000 version pairs
"""

from __future__ import annotations

import time

from retrieval_os.serving.cache import _cache_key

# ── Throughput ────────────────────────────────────────────────────────────────


class TestCacheKeyThroughput:
    def test_100k_keys_under_2s(self, record_bm) -> None:
        """100 000 cache key generations must complete in < 2 seconds.

        At 10 000 QPS each request generates one cache key. 100 000 keys
        represents 10 seconds of peak traffic; key generation must not exceed
        2 seconds of CPU time (20% overhead budget).
        """
        inputs = [
            (f"project-{i % 100}", i % 10, f"query text number {i} about topic {i % 500}", 10)
            for i in range(100_000)
        ]

        start = time.perf_counter()
        for project, version, query, top_k in inputs:
            _cache_key(project, version, query, top_k)
        elapsed = time.perf_counter() - start

        record_bm(
            "100k cache key generations (SHA-256)", elapsed, limit_s=2.0, n=100_000, unit="key"
        )
        assert elapsed < 2.0, (
            f"100 000 cache key generations took {elapsed:.3f}s; must be < 2s "
            f"({elapsed / 100_000 * 1_000:.4f} ms/key)"
        )


# ── Collision resistance ──────────────────────────────────────────────────────


class TestCacheKeyCollisionResistance:
    def test_100k_distinct_inputs_produce_100k_distinct_keys(self) -> None:
        """Zero collisions across 100 000 distinct cache key inputs.

        A collision would cause wrong cached results to be served for a query,
        silently corrupting retrieval quality metrics and end-user results.
        """
        inputs = [
            (f"project-{i % 200}", i % 20, f"unique query {i}: what is approach {i}?", i % 5 + 1)
            for i in range(100_000)
        ]

        keys = [_cache_key(p, v, q, k) for p, v, q, k in inputs]
        unique_keys = set(keys)

        assert len(unique_keys) == 100_000, (
            f"Expected 100 000 unique keys; got {len(unique_keys)} "
            f"({100_000 - len(unique_keys)} collisions)"
        )

    def test_keys_are_64_hex_chars(self) -> None:
        """Every key is a 64-character lowercase hex string (SHA-256 digest)."""
        keys = [_cache_key(f"proj-{i}", i, f"query {i}", 10) for i in range(1_000)]
        for key in keys:
            # Strip the prefix to get the digest
            digest = key.split(":")[-1]
            assert len(digest) == 64, f"Digest length is {len(digest)}, expected 64"
            assert all(c in "0123456789abcdef" for c in digest), f"Non-hex characters in key: {key}"


# ── Isolation guarantees ──────────────────────────────────────────────────────


class TestCacheKeyIsolation:
    def test_same_query_different_projects_never_share_key(self) -> None:
        """The same query text on different projects must produce different cache keys.

        Without this, Project A's cached results could be served to Project B —
        a critical tenant data isolation failure.
        """
        query = "what is the best embedding model for semantic search?"
        top_k = 10
        version = 1

        project_keys = {
            f"project-{i}": _cache_key(f"project-{i}", version, query, top_k) for i in range(1_000)
        }
        unique_keys = set(project_keys.values())

        assert len(unique_keys) == 1_000, (
            f"Same query on different projects produced duplicate keys. "
            f"Unique keys: {len(unique_keys)} for 1 000 projects."
        )

    def test_same_query_different_versions_never_share_key(self) -> None:
        """The same query on the same project but different index versions must produce
        different keys.

        Without this, results cached from an old index version would be served
        after a new deployment, bypassing the new embedding model entirely.
        """
        project = "docs"
        query = "how does chunking affect retrieval quality?"
        top_k = 10

        version_keys = {
            version: _cache_key(project, version, query, top_k) for version in range(1, 1_001)
        }
        unique_keys = set(version_keys.values())

        assert len(unique_keys) == 1_000, (
            f"Same query at different index versions produced duplicate keys. "
            f"Unique keys: {len(unique_keys)} for 1 000 versions."
        )

    def test_same_query_different_top_k_never_share_key(self) -> None:
        """Different top_k values must produce different keys.

        A query with top_k=5 and top_k=20 return different result sets.
        Sharing a key would serve 5 results when 20 were requested (or vice versa).
        """
        project = "search"
        version = 1
        query = "retrieval augmented generation"

        top_k_keys = {k: _cache_key(project, version, query, k) for k in range(1, 101)}
        unique_keys = set(top_k_keys.values())

        assert len(unique_keys) == 100, (
            f"Different top_k values produced duplicate keys. "
            f"Unique keys: {len(unique_keys)} for 100 top_k values."
        )
