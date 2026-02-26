"""Benchmark: Iteration Velocity

Customer claim: teams can iterate on retrieval config (embed model, chunking,
distance metric) and get a new indexed version live without a multi-hour
reindex cycle blocking them.

What we measure
---------------
The three operations that sit on the critical path of every deploy:

1. Config hash computation — every new IndexConfig is hashed before it is
   written. At scale (CI pipelines running hundreds of config candidates), this
   must not become a bottleneck.

2. Config validation — runs at write time so bad configs are rejected before
   any embedding work starts. Validation must be fast enough to check many
   candidate configs in a single CI run.

3. Hash deduplication correctness at scale — the system must detect
   identical configs across thousands of concurrent experiments with zero
   false positives and zero false negatives.

Scale targets
-------------
- 10 000 config hashes   in < 2 s   → < 0.2 ms per hash
- 1 000 config validations in < 1 s → < 1 ms per validation
- 10 000 distinct configs → 10 000 distinct hashes (zero collisions)
- N configs with identical index fields but different search fields → one
  unique hash (all search-field variants collapse to the same hash)
"""

from __future__ import annotations

import time

from retrieval_os.plans.validators import compute_config_hash, validate_index_config

# ── Synthetic config factories ────────────────────────────────────────────────


def _index_config(
    *,
    model: str = "BAAI/bge-m3",
    collection: str = "docs_v1",
    provider: str = "sentence_transformers",
    dimensions: int = 768,
    metric: str = "cosine",
) -> dict:
    return {
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimensions": dimensions,
        "modalities": ["text"],
        "embedding_batch_size": 32,
        "embedding_normalize": True,
        "index_backend": "qdrant",
        "index_collection": collection,
        "distance_metric": metric,
        "quantization": None,
        "change_comment": "",
    }


def _config_with_search_fields(**search_overrides: object) -> dict:
    """Index config with search-config fields that should NOT affect the hash."""
    base = _index_config()
    base.update(search_overrides)
    return base


# ── Throughput: config hash computation ───────────────────────────────────────


class TestConfigHashThroughput:
    def test_10k_hashes_under_2s(self) -> None:
        """10 000 config hash computations must complete in under 2 seconds.

        At this rate a CI pipeline can evaluate hundreds of config candidates
        per second without hash computation becoming the bottleneck.
        """
        configs = [
            _index_config(
                model=f"model-{i % 50}",
                collection=f"col-{i % 20}",
                dimensions=768 + (i % 4) * 256,
            )
            for i in range(10_000)
        ]

        start = time.perf_counter()
        for cfg in configs:
            compute_config_hash(cfg)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"10 000 config hashes took {elapsed:.3f}s; must be < 2s "
            f"({elapsed / 10_000 * 1000:.3f} ms/hash)"
        )


# ── Throughput: config validation ─────────────────────────────────────────────


class TestConfigValidationThroughput:
    def test_1k_validations_under_1s(self) -> None:
        """1 000 config validations must complete in under 1 second.

        Config validation is the first gate before any embedding work starts.
        At 1 ms/validation, a sweep of 1 000 candidate configs (common in
        hyper-parameter search) completes before the first embedding job runs.
        """
        configs = [
            _index_config(
                model=f"model-{i % 10}",
                collection=f"col-{i % 5}",
            )
            for i in range(1_000)
        ]

        start = time.perf_counter()
        for cfg in configs:
            validate_index_config(cfg)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"1 000 config validations took {elapsed:.3f}s; must be < 1s "
            f"({elapsed / 1_000 * 1000:.3f} ms/validation)"
        )


# ── Deduplication correctness at scale ────────────────────────────────────────


class TestHashDeduplicationAtScale:
    def test_10k_distinct_configs_produce_10k_distinct_hashes(self) -> None:
        """Zero hash collisions across 10 000 distinct index configurations.

        Each unique (model, collection, dimensions) triple must produce a
        unique hash. A collision would cause a valid new config to be silently
        rejected as a duplicate.
        """
        configs = [
            _index_config(
                model=f"provider-{i // 100}/model-{i % 100}",
                collection=f"collection-v{i}",
                dimensions=128 + (i % 10) * 128,
            )
            for i in range(10_000)
        ]

        hashes = [compute_config_hash(c) for c in configs]
        unique_hashes = set(hashes)

        assert len(unique_hashes) == 10_000, (
            f"Expected 10 000 unique hashes; got {len(unique_hashes)} "
            f"({10_000 - len(unique_hashes)} collisions)"
        )

    def test_search_field_variants_collapse_to_one_hash(self) -> None:
        """All search-config variants of the same index config share one hash.

        A team tuning top_k, reranker, cache_ttl, hybrid_alpha should NOT
        generate thousands of 'new' index configs — they all map to the same
        index and the same hash.
        """
        search_variants = [
            {"top_k": k, "reranker": r, "cache_ttl_seconds": t, "hybrid_alpha": a}
            for k in [5, 10, 20, 50, 100]
            for r in [None, "cross-encoder/ms-marco-MiniLM-L-6-v2"]
            for t in [0, 300, 900, 3600]
            for a in [None, 0.3, 0.5, 0.7, 1.0]
        ]
        # 5 × 2 × 4 × 5 = 200 variants of the same underlying index config

        hashes = {compute_config_hash(_config_with_search_fields(**v)) for v in search_variants}

        assert len(hashes) == 1, (
            f"Expected 1 unique hash for 200 search-config variants; "
            f"got {len(hashes)} (search fields incorrectly included in hash)"
        )

    def test_single_field_change_always_changes_hash(self) -> None:
        """Every distinct index-build field value produces a distinct hash.

        Regression guard: no two meaningfully different index configs should
        share a hash, which would allow one to shadow the other.
        """
        base = _index_config()
        base_hash = compute_config_hash(base)

        variants = [
            ("embedding_model", "text-embedding-3-large"),
            ("embedding_model", "text-embedding-3-small"),
            ("embedding_dimensions", 1536),
            ("embedding_dimensions", 256),
            ("index_collection", "docs_v2"),
            ("index_collection", "docs_production"),
            ("distance_metric", "dot"),
            ("distance_metric", "euclidean"),
            ("index_backend", "pgvector"),
            ("quantization", "scalar"),
            ("quantization", "product"),
        ]

        seen: set[str] = {base_hash}
        for field, value in variants:
            cfg = {**base, field: value}
            h = compute_config_hash(cfg)
            assert h not in seen, (
                f"Hash collision: changing {field}={value!r} produced the same hash "
                "as a previous config"
            )
            seen.add(h)
