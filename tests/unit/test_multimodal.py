"""Unit tests for multimodal embedding helpers.

These tests validate the module-level API contracts and error handling
without loading any actual ML models (open_clip, faster_whisper, torch).
Model loading is deferred to first-use behind try/ImportError guards.
"""

from __future__ import annotations

import asyncio

import pytest

from retrieval_os.core.exceptions import EmbeddingProviderError

# ── encode_images_clip ────────────────────────────────────────────────────────


class TestEncodeImagesClip:
    @pytest.mark.asyncio
    async def test_raises_provider_error_without_open_clip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """simulate open_clip not installed → EmbeddingProviderError."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "open_clip":
                raise ImportError("No module named 'open_clip'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Clear cached singleton so it re-imports
        import retrieval_os.serving.multimodal as mm

        mm._clip_state.clear()

        with pytest.raises(EmbeddingProviderError, match="open-clip-torch"):
            await mm.encode_images_clip([b"fake-image-bytes"])

        mm._clip_state.clear()

    def test_function_is_coroutine(self) -> None:
        from retrieval_os.serving.multimodal import encode_images_clip

        assert asyncio.iscoroutinefunction(encode_images_clip)


# ── transcribe_audio_whisper ──────────────────────────────────────────────────


class TestTranscribeAudioWhisper:
    @pytest.mark.asyncio
    async def test_raises_provider_error_without_faster_whisper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """simulate faster_whisper not installed → EmbeddingProviderError."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "faster_whisper":
                raise ImportError("No module named 'faster_whisper'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        import retrieval_os.serving.multimodal as mm

        mm._whisper_state.clear()

        with pytest.raises(EmbeddingProviderError, match="faster-whisper"):
            await mm.transcribe_audio_whisper(b"fake-audio-bytes")

        mm._whisper_state.clear()

    def test_function_is_coroutine(self) -> None:
        from retrieval_os.serving.multimodal import transcribe_audio_whisper

        assert asyncio.iscoroutinefunction(transcribe_audio_whisper)


# ── embed_images in embed_router ─────────────────────────────────────────────


class TestEmbedImagesRouter:
    def test_function_exists(self) -> None:
        from retrieval_os.serving.embed_router import embed_images

        assert asyncio.iscoroutinefunction(embed_images)

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self) -> None:
        from retrieval_os.serving.embed_router import embed_images

        with pytest.raises(EmbeddingProviderError, match="Unknown image embedding provider"):
            await embed_images([b"img"], provider="unknown", model="x")

    @pytest.mark.asyncio
    async def test_video_frame_raises_not_implemented(self) -> None:
        from retrieval_os.serving.embed_router import embed_images

        with pytest.raises(EmbeddingProviderError, match="video_frame"):
            await embed_images([b"img"], provider="video_frame", model="x")


# ── embed_audio in embed_router ───────────────────────────────────────────────


class TestEmbedAudioRouter:
    def test_function_exists(self) -> None:
        from retrieval_os.serving.embed_router import embed_audio

        assert asyncio.iscoroutinefunction(embed_audio)


# ── embed_text updated stubs ──────────────────────────────────────────────────


class TestEmbedTextUpdatedStubs:
    @pytest.mark.asyncio
    async def test_clip_via_embed_text_redirects_user(self) -> None:
        """Calling embed_text with provider='clip' should give a helpful error."""
        from retrieval_os.serving.embed_router import embed_text

        with pytest.raises(EmbeddingProviderError, match="embed_images"):
            await embed_text(["some text"], provider="clip", model="ViT-B-32")

    @pytest.mark.asyncio
    async def test_whisper_via_embed_text_redirects_user(self) -> None:
        """Calling embed_text with provider='whisper' should give a helpful error."""
        from retrieval_os.serving.embed_router import embed_text

        with pytest.raises(EmbeddingProviderError, match="embed_audio"):
            await embed_text(["some text"], provider="whisper", model="base")
