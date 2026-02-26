"""Load test: Multimodal query throughput — image (CLIP) and audio (Whisper) paths.

What this proves
----------------
1. Image query p99 with CLIP stubbed — measures config-load + Qdrant ANN overhead
   for the image path (identical infrastructure to text, so numbers should be
   comparable to the cache-miss baseline in test_query_latency.py).
2. Audio query p99 with Whisper + text-embed both stubbed — same infra measurement
   for the two-stage audio pipeline.
3. Concurrent image and concurrent audio bursts — QPS under realistic concurrency.
4. Whisper transcription latency overhead — sleep stubs calibrated to tiny / base /
   large model inference times quantify what each Whisper size adds before the
   embedding step.
5. Mixed text + image concurrency — verifies that the two modality paths do not
   interfere with each other on shared Qdrant + Redis.

Patch targets
-------------
- ``retrieval_os.serving.query_router.embed_images``   → stubs CLIP for route_image_query
- ``retrieval_os.serving.query_router.embed_audio``    → stubs full audio pipeline for route_audio_query
- ``retrieval_os.serving.multimodal.transcribe_audio_whisper``  → stubs only Whisper (leaves
  embed_text real) for the transcription-overhead tests
- ``retrieval_os.serving.embed_router.embed_text``     → stubs text embed within embed_audio
  for the transcription-overhead tests

Infrastructure required: Postgres + Redis + Qdrant.
Auto-skipped when any service is unreachable.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

from retrieval_os.core.database import async_session_factory
from retrieval_os.core.redis_client import get_redis
from retrieval_os.deployments.schemas import CreateDeploymentRequest
from retrieval_os.deployments.service import create_deployment
from retrieval_os.plans.schemas import CreateProjectRequest, IndexConfigInput
from retrieval_os.plans.service import create_project
from retrieval_os.serving.embed_router import embed_audio
from retrieval_os.serving.query_router import route_audio_query, route_image_query
from tests.load.conftest import DIMS, random_unit_vector

# ── Constants ──────────────────────────────────────────────────────────────────

# Dummy bytes — content irrelevant; real CLIP / Whisper models are stubbed.
_FAKE_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 64  # JPEG-ish header + padding
_FAKE_AUDIO = b"RIFF" + b"\x00" * 64  # WAV-ish header + padding

# ── Helpers ────────────────────────────────────────────────────────────────────


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[min(int(len(s) * p), len(s) - 1)]


# ── Clip project fixture ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def clip_project(load_collection, check_load_infra) -> str:  # type: ignore[misc]
    """Create a project wired to the CLIP provider pointing at the shared collection.

    embedding_dimensions=DIMS (384) intentionally matches the shared 384-dim
    collection even though real ViT-B-32 produces 512-d vectors.  The stub
    returns DIMS-d vectors, so Qdrant never sees a dimension mismatch.
    """
    name = f"load-clip-{uuid.uuid4().hex[:8]}"

    async with async_session_factory() as session:
        await create_project(
            session,
            CreateProjectRequest(
                name=name,
                description="Multimodal load test — CLIP image",
                config=IndexConfigInput(
                    embedding_provider="clip",
                    embedding_model="ViT-B-32",
                    embedding_dimensions=DIMS,
                    modalities=["image"],
                    index_collection=load_collection,
                    index_backend="qdrant",
                    distance_metric="cosine",
                    change_comment="multimodal load test",
                ),
                created_by="load-test",
            ),
        )
        await session.commit()

    async with async_session_factory() as session:
        await create_deployment(
            session,
            name,
            CreateDeploymentRequest(
                index_config_version=1,
                top_k=10,
                cache_enabled=False,  # multimodal routes never hit the query cache
                cache_ttl_seconds=3600,
                created_by="load-test",
            ),
        )
        await session.commit()

    yield name

    # Best-effort teardown
    try:
        redis = await get_redis()
        await redis.delete(
            f"ros:project:{name}:active",
            f"ros:deployment:{name}:active",
        )
    except Exception:
        pass
    try:
        async with async_session_factory() as session:
            await session.execute(
                text("DELETE FROM deployments WHERE project_name = :n"), {"n": name}
            )
            await session.execute(
                text(
                    "DELETE FROM index_configs "
                    "WHERE project_id = (SELECT id FROM projects WHERE name = :n)"
                ),
                {"n": name},
            )
            await session.execute(text("DELETE FROM projects WHERE name = :n"), {"n": name})
            await session.commit()
    except Exception:
        pass


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestImageQueryThroughput:
    """CLIP image query path: config-load + Qdrant ANN, embed stubbed."""

    async def test_image_query_p99_baseline(self, clip_project, record_load) -> None:
        """50 sequential image queries with CLIP stubbed — p99 < 50 ms.

        The serving infrastructure (Redis config-load + Qdrant ANN) is
        identical to the text cache-miss path.  This test confirms it adds
        no overhead specific to the image modality routing.
        """
        stub_vector = random_unit_vector()
        n_queries = 50
        latencies: list[float] = []

        with patch(
            "retrieval_os.serving.query_router.embed_images",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            async with async_session_factory() as session:
                for i in range(n_queries):
                    t0 = time.perf_counter()
                    await route_image_query(
                        project_name=clip_project,
                        image_bytes=_FAKE_IMAGE,
                        db=session,
                    )
                    latencies.append((time.perf_counter() - t0) * 1000)

        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Image query baseline (CLIP stubbed, 50 sequential)",
            samples=latencies,
            qps=qps,
            note=f"p99={p99:.1f}ms",
        )

        assert p99 < 50.0, (
            f"Image query p99={p99:.1f}ms; expected < 50 ms. "
            "Multimodal routing should add no overhead beyond text cache-miss path."
        )

    async def test_concurrent_image_query_burst(self, clip_project, record_load) -> None:
        """20 concurrent image queries — QPS ≥ 20, zero errors.

        Each worker uses its own DB session (cheap; Redis is warm after
        the baseline test so the session is not actually used for queries).
        """
        stub_vector = random_unit_vector()
        n_workers = 20
        n_per_worker = 5
        errors: list[Exception] = []
        latencies: list[float] = []

        async def _worker() -> None:
            async with async_session_factory() as session:
                with patch(
                    "retrieval_os.serving.query_router.embed_images",
                    new_callable=AsyncMock,
                    return_value=[stub_vector],
                ):
                    for _ in range(n_per_worker):
                        try:
                            t0 = time.perf_counter()
                            await route_image_query(
                                project_name=clip_project,
                                image_bytes=_FAKE_IMAGE,
                                db=session,
                            )
                            latencies.append((time.perf_counter() - t0) * 1000)
                        except Exception as exc:
                            errors.append(exc)

        wall_start = time.perf_counter()
        # Patch outside the gather to avoid per-worker patch race condition
        with patch(
            "retrieval_os.serving.query_router.embed_images",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            await asyncio.gather(*[_worker() for _ in range(n_workers)])
        wall_elapsed = time.perf_counter() - wall_start

        assert errors == [], f"{len(errors)} errors during concurrent image burst: {errors[:3]}"

        total = n_workers * n_per_worker
        qps = total / wall_elapsed

        record_load(
            f"Image query burst ({n_workers} concurrent workers)",
            samples=latencies,
            qps=qps,
            note=f"n={total}, errors=0",
        )

        assert qps >= 20.0, (
            f"Concurrent image query QPS={qps:.0f}; expected ≥ 20. "
            "Check Qdrant ANN latency or asyncio contention."
        )


class TestAudioQueryThroughput:
    """Audio query path: Whisper + text-embed both stubbed, Qdrant ANN real."""

    async def test_audio_query_p99_baseline(
        self, load_project, load_collection, record_load
    ) -> None:
        """50 sequential audio queries with embed_audio fully stubbed — p99 < 50 ms.

        Uses the standard text project (embedding_provider="sentence_transformers").
        route_audio_query passes text_provider="sentence_transformers" to embed_audio,
        which calls transcribe_audio_whisper → embed_text.  Both are stubbed here so
        only the infrastructure layer is measured.
        """
        stub_vector = random_unit_vector()
        n_queries = 50
        latencies: list[float] = []

        with patch(
            "retrieval_os.serving.query_router.embed_audio",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            async with async_session_factory() as session:
                for i in range(n_queries):
                    t0 = time.perf_counter()
                    await route_audio_query(
                        project_name=load_project,
                        audio_bytes=_FAKE_AUDIO,
                        db=session,
                    )
                    latencies.append((time.perf_counter() - t0) * 1000)

        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Audio query baseline (embed_audio stubbed, 50 sequential)",
            samples=latencies,
            qps=qps,
            note=f"p99={p99:.1f}ms",
        )

        assert p99 < 50.0, (
            f"Audio query p99={p99:.1f}ms; expected < 50 ms. "
            "Audio routing should add no overhead beyond text cache-miss path."
        )

    async def test_concurrent_audio_query_burst(self, load_project, record_load) -> None:
        """20 concurrent audio queries — QPS ≥ 20, zero errors."""
        stub_vector = random_unit_vector()
        n_workers = 20
        n_per_worker = 5
        errors: list[Exception] = []
        latencies: list[float] = []

        async def _worker() -> None:
            async with async_session_factory() as session:
                for _ in range(n_per_worker):
                    try:
                        t0 = time.perf_counter()
                        await route_audio_query(
                            project_name=load_project,
                            audio_bytes=_FAKE_AUDIO,
                            db=session,
                        )
                        latencies.append((time.perf_counter() - t0) * 1000)
                    except Exception as exc:
                        errors.append(exc)

        wall_start = time.perf_counter()
        with patch(
            "retrieval_os.serving.query_router.embed_audio",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            await asyncio.gather(*[_worker() for _ in range(n_workers)])
        wall_elapsed = time.perf_counter() - wall_start

        assert errors == [], f"{len(errors)} errors during concurrent audio burst: {errors[:3]}"

        total = n_workers * n_per_worker
        qps = total / wall_elapsed

        record_load(
            f"Audio query burst ({n_workers} concurrent workers)",
            samples=latencies,
            qps=qps,
            note=f"n={total}, errors=0",
        )

        assert qps >= 20.0, f"Concurrent audio query QPS={qps:.0f}; expected ≥ 20."


class TestWhisperTranscriptionOverhead:
    """Additive latency of Whisper transcription at different model sizes.

    Calls embed_audio() directly (no DB / project needed) with transcription
    stubbed as a calibrated sleep.  embed_text is also stubbed so the only
    measured overhead is the sleep — verifying the pipeline wires correctly and
    the overhead is predictable.

    Reference inference times on CPU (rough estimates):
      tiny   ~30 ms   (39 M params)
      base   ~150 ms  (74 M params)
      large  ~500 ms  (1.5 B params)
    """

    async def _run_overhead_test(
        self,
        label: str,
        transcription_latency_s: float,
        n_queries: int,
        record_load,
    ) -> tuple[float, float]:
        """Run n_queries through embed_audio with a calibrated transcription stub."""
        stub_vector = random_unit_vector()
        latencies: list[float] = []

        async def _fake_transcribe(audio_bytes: bytes, *, model_size: str = "base", **_) -> str:
            await asyncio.sleep(transcription_latency_s)
            return "stub transcript from whisper"

        with (
            patch(
                "retrieval_os.serving.multimodal.transcribe_audio_whisper",
                side_effect=_fake_transcribe,
            ),
            patch(
                "retrieval_os.serving.embed_router.embed_text",
                new_callable=AsyncMock,
                return_value=[stub_vector],
            ),
        ):
            for i in range(n_queries):
                t0 = time.perf_counter()
                await embed_audio(
                    [_FAKE_AUDIO],
                    whisper_model_size="base",
                    text_provider="sentence_transformers",
                    text_model="all-MiniLM-L6-v2",
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        p50 = _percentile(latencies, 0.50)
        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            label,
            samples=latencies,
            qps=qps,
            note=f"transcription_stub={transcription_latency_s * 1000:.0f}ms, p50={p50:.1f}ms",
        )
        return p50, p99

    async def test_whisper_tiny_overhead(self, record_load) -> None:
        """Tiny model (~30 ms): total p50 should exceed the stub latency."""
        overhead_s = 0.030
        p50, p99 = await self._run_overhead_test(
            "Whisper overhead — tiny (~30 ms stub)",
            transcription_latency_s=overhead_s,
            n_queries=20,
            record_load=record_load,
        )
        assert p50 > overhead_s * 1000 * 0.9, (
            f"p50={p50:.1f}ms should exceed {overhead_s * 1000:.0f}ms stub. "
            "Check transcription stub is actually running."
        )

    async def test_whisper_base_overhead(self, record_load) -> None:
        """Base model (~150 ms): total p50 should exceed stub, p99 bounded."""
        overhead_s = 0.150
        p50, p99 = await self._run_overhead_test(
            "Whisper overhead — base (~150 ms stub)",
            transcription_latency_s=overhead_s,
            n_queries=20,
            record_load=record_load,
        )
        assert p50 > overhead_s * 1000 * 0.9, (
            f"p50={p50:.1f}ms should exceed {overhead_s * 1000:.0f}ms stub."
        )
        assert p99 < overhead_s * 1000 + 100, (
            f"p99={p99:.1f}ms; expected < {overhead_s * 1000 + 100:.0f}ms. "
            "Transcription overhead is inconsistent."
        )

    async def test_whisper_large_overhead(self, record_load) -> None:
        """Large model (~500 ms): total p50 should exceed stub, p99 bounded."""
        overhead_s = 0.500
        p50, p99 = await self._run_overhead_test(
            "Whisper overhead — large (~500 ms stub)",
            transcription_latency_s=overhead_s,
            n_queries=10,
            record_load=record_load,
        )
        assert p50 > overhead_s * 1000 * 0.9, (
            f"p50={p50:.1f}ms should exceed {overhead_s * 1000:.0f}ms stub."
        )
        assert p99 < overhead_s * 1000 + 200, (
            f"p99={p99:.1f}ms; expected < {overhead_s * 1000 + 200:.0f}ms."
        )


class TestMixedModalityTraffic:
    """Text and image queries share Qdrant + Redis without interfering."""

    async def test_concurrent_text_and_image_no_interference(
        self, load_project, load_collection, clip_project, record_load
    ) -> None:
        """10 text workers + 10 image workers run simultaneously.

        Asserts:
        - Zero errors across both modalities.
        - Neither modality's p99 exceeds 50 ms (single-node ANN SLA).
        """
        from retrieval_os.serving.executor import execute_retrieval

        stub_vector = random_unit_vector()
        n_workers_each = 10
        n_per_worker = 5

        text_latencies: list[float] = []
        image_latencies: list[float] = []
        errors: list[Exception] = []

        def _text_exec_kwargs(query: str) -> dict:
            return dict(
                project_name=load_project,
                version=1,
                query=query,
                embedding_provider="sentence_transformers",
                embedding_model="all-MiniLM-L6-v2",
                embedding_normalize=True,
                embedding_batch_size=32,
                index_backend="qdrant",
                index_collection=load_collection,
                distance_metric="cosine",
                top_k=10,
                reranker=None,
                rerank_top_k=None,
                metadata_filters=None,
                cache_enabled=False,
                cache_ttl_seconds=3600,
            )

        async def _text_worker(worker_id: int) -> None:
            for qi in range(n_per_worker):
                try:
                    t0 = time.perf_counter()
                    await execute_retrieval(
                        **_text_exec_kwargs(f"mixed text worker {worker_id} q{qi}")
                    )
                    text_latencies.append((time.perf_counter() - t0) * 1000)
                except Exception as exc:
                    errors.append(exc)

        async def _image_worker() -> None:
            async with async_session_factory() as session:
                for _ in range(n_per_worker):
                    try:
                        t0 = time.perf_counter()
                        await route_image_query(
                            project_name=clip_project,
                            image_bytes=_FAKE_IMAGE,
                            db=session,
                        )
                        image_latencies.append((time.perf_counter() - t0) * 1000)
                    except Exception as exc:
                        errors.append(exc)

        with (
            patch(
                "retrieval_os.serving.executor.embed_text",
                new_callable=AsyncMock,
                return_value=[stub_vector],
            ),
            patch(
                "retrieval_os.serving.query_router.embed_images",
                new_callable=AsyncMock,
                return_value=[stub_vector],
            ),
        ):
            await asyncio.gather(
                *[_text_worker(i) for i in range(n_workers_each)],
                *[_image_worker() for _ in range(n_workers_each)],
            )

        assert errors == [], f"{len(errors)} errors during mixed traffic: {errors[:3]}"

        text_p99 = _percentile(text_latencies, 0.99)
        image_p99 = _percentile(image_latencies, 0.99)

        text_qps_approx = (
            len(text_latencies) / (sum(text_latencies) / 1000) if text_latencies else 0
        )
        image_qps_approx = (
            len(image_latencies) / (sum(image_latencies) / 1000) if image_latencies else 0
        )

        record_load(
            f"Mixed modality — text ({n_workers_each} workers)",
            samples=text_latencies,
            qps=text_qps_approx,
            note=f"concurrent with {n_workers_each} image workers",
        )
        record_load(
            f"Mixed modality — image ({n_workers_each} workers)",
            samples=image_latencies,
            qps=image_qps_approx,
            note=f"concurrent with {n_workers_each} text workers",
        )

        assert text_p99 < 50.0, (
            f"Text p99={text_p99:.1f}ms under mixed load exceeds 50 ms. "
            "Image workers may be starving text path."
        )
        assert image_p99 < 50.0, (
            f"Image p99={image_p99:.1f}ms under mixed load exceeds 50 ms. "
            "Text workers may be starving image path."
        )
