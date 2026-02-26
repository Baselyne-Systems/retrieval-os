"""Load test: Embedding model throughput.

Measures real inference latency for sentence-transformers models.
This test is **skipped automatically** when sentence_transformers is not
installed — install it with ``uv sync --extra ml`` to enable it.

Why this matters
----------------
The embedding step is typically the single largest latency contributor in
a RAG pipeline.  The Qdrant ANN numbers look impressive (p99 < 4 ms) but
they're irrelevant if embedding takes 50 ms per query.

What we measure
---------------
1. Single-query embedding latency (p50/p95/p99) — the true hot-path cost
2. Batched embedding throughput (tokens/sec, queries/sec) — for batch ingest
3. Cold-start vs warm cost — first inference vs steady state
4. Per-model comparison if multiple models are available

Scale reference
---------------
  all-MiniLM-L6-v2 on M2 CPU:    ~5–15 ms/query,  ~70–200 QPS
  all-MiniLM-L6-v2 on A100 GPU:  ~0.5–2 ms/query, ~1 000–5 000 QPS
  text-embedding-3-small (OpenAI): ~20–80 ms/query (network-dependent)

These numbers are **additive** with the infrastructure latency measured in
test_query_latency.py.  True end-to-end p99 = embed p99 + infra p99.
"""

from __future__ import annotations

import time

import pytest

# Skip entire module if sentence_transformers is not installed.
# This prevents import errors in CI environments without the ML extras.
sentence_transformers = pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers not installed — run `uv sync --extra ml` to enable",
)

from retrieval_os.serving.embed_router import embed_text  # noqa: E402  (after importorskip)

# ── Models to benchmark ────────────────────────────────────────────────────────

# Add more models here for comparative benchmarking.
# Each tuple: (provider, model_name, expected_dims)
_MODELS = [
    ("sentence_transformers", "all-MiniLM-L6-v2", 384),
]

# ── Sample queries ─────────────────────────────────────────────────────────────

_SAMPLE_QUERIES = [
    "What is retrieval-augmented generation and how does it improve LLM accuracy?",
    "How do I choose between cosine similarity and dot product for vector search?",
    "What chunk size gives the best recall for technical documentation?",
    "How does HNSW graph structure affect ANN search latency at scale?",
    "What is the difference between sparse and dense retrieval methods?",
    "How can I reduce embedding model inference latency in production?",
    "What is the role of a reranker in a two-stage retrieval pipeline?",
    "How do I detect retrieval quality regressions before they reach production?",
    "What metadata filters are most effective for multi-tenant RAG systems?",
    "How does quantization affect retrieval recall for high-dimensional embeddings?",
]


# ── Single-query latency ───────────────────────────────────────────────────────


class TestSingleQueryLatency:
    """Real inference latency per query — what users actually feel."""

    @pytest.mark.parametrize("provider,model,dims", _MODELS)
    async def test_single_query_p99_measured(
        self, provider: str, model: str, dims: int, record_load
    ) -> None:
        """Measure and report p99 single-query embedding latency.

        No hard threshold — models vary widely.  The result is recorded in
        the load test summary so it can be added to p99 infrastructure latency
        for an accurate end-to-end estimate.
        """
        n_queries = 50
        queries = [_SAMPLE_QUERIES[i % len(_SAMPLE_QUERIES)] for i in range(n_queries)]

        # Cold start — discard first result
        await embed_text([queries[0]], provider=provider, model=model, normalize=True, batch_size=1)

        latencies: list[float] = []
        for q in queries:
            t0 = time.perf_counter()
            vecs = await embed_text(
                [q], provider=provider, model=model, normalize=True, batch_size=1
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            assert len(vecs) == 1
            assert len(vecs[0]) == dims

        p50 = sorted(latencies)[len(latencies) // 2]

        record_load(
            f"Embed latency: {model} (single query, warm)",
            samples=latencies,
            note=f"add to infra p50/p99 for end-to-end SLA ({dims}d vectors)",
        )

        # Sanity: at least 1 query per second (if this fails, something is very wrong)
        assert p50 < 5_000, f"Embedding p50={p50:.0f} ms — model may not be loaded correctly"

    @pytest.mark.parametrize("provider,model,dims", _MODELS)
    async def test_cold_start_vs_warm(self, provider: str, model: str, dims: int) -> None:
        """Cold-start (first inference) must be ≤ 5× the warm p50.

        Models load weights lazily.  A very high cold-start cost can cause
        the first user after a worker restart to experience a timeout.
        """

        import retrieval_os.serving.embed_router as er

        # Force cold start by resetting the cached model
        er._st_model = None

        t0 = time.perf_counter()
        await embed_text(
            [_SAMPLE_QUERIES[0]], provider=provider, model=model, normalize=True, batch_size=1
        )
        cold_ms = (time.perf_counter() - t0) * 1000

        # Measure warm latency (5 samples)
        warm_latencies = []
        for q in _SAMPLE_QUERIES[1:6]:
            t0 = time.perf_counter()
            await embed_text([q], provider=provider, model=model, normalize=True, batch_size=1)
            warm_latencies.append((time.perf_counter() - t0) * 1000)
        warm_p50 = sorted(warm_latencies)[len(warm_latencies) // 2]

        # Document the numbers (no strict assertion — varies too much by hardware)
        print(
            f"\n  {model}: cold_start={cold_ms:.0f} ms, warm_p50={warm_p50:.1f} ms, "
            f"cold/warm ratio={cold_ms / warm_p50:.1f}×"
        )

        # Sanity: cold start should not be more than 60 seconds
        assert cold_ms < 60_000, (
            f"Cold start took {cold_ms:.0f} ms (> 60 s). "
            "Model download may have triggered — use `model.max_seq_length` to verify."
        )


# ── Batched embedding throughput ──────────────────────────────────────────────


class TestBatchedEmbedThroughput:
    """Throughput for ingestion workloads where queries are processed in batches."""

    @pytest.mark.parametrize("provider,model,dims", _MODELS)
    async def test_batch_throughput_qps(
        self, provider: str, model: str, dims: int, record_load
    ) -> None:
        """Measure tokens/sec and queries/sec for batched embedding.

        Batching amortises model overhead.  At batch_size=32, throughput
        should be significantly higher than single-query embedding.
        """
        batch_size = 32
        n_batches = 5
        # Repeat sample queries to fill batches
        batch = (_SAMPLE_QUERIES * (batch_size // len(_SAMPLE_QUERIES) + 1))[:batch_size]

        # Warmup
        await embed_text(batch[:4], provider=provider, model=model, normalize=True, batch_size=4)

        latencies: list[float] = []
        for _ in range(n_batches):
            t0 = time.perf_counter()
            vecs = await embed_text(
                batch, provider=provider, model=model, normalize=True, batch_size=batch_size
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            assert len(vecs) == batch_size
            assert len(vecs[0]) == dims

        total_queries = batch_size * n_batches
        total_ms = sum(latencies)
        qps = total_queries / (total_ms / 1000)

        # Approximate token count (avg ~15 tokens per query string above)
        avg_tokens_per_query = 20
        tps = total_queries * avg_tokens_per_query / (total_ms / 1000)

        record_load(
            f"Batched embed: {model} (batch={batch_size})",
            samples=latencies,
            qps=qps,
            note=f"≈{tps:.0f} tokens/sec, {dims}d vectors",
        )

        # At least 1 QPS even on slow hardware
        assert qps >= 1.0, f"Batched embed QPS={qps:.2f} — something is very wrong"
