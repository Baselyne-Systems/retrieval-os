"""Multimodal embedding helpers — CLIP (image→vector) and Whisper (audio→transcript).

Both helpers are run in asyncio's thread pool so they never block the event loop.
ML dependencies (open_clip, faster_whisper, torch, Pillow) are deferred to
first-use so the API starts fast even without the `ml` extras installed.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from retrieval_os.core.exceptions import EmbeddingProviderError

log = logging.getLogger(__name__)

# ── Lazy singletons ────────────────────────────────────────────────────────────
# Keyed by model identifier so different models can coexist if needed.

_clip_state: dict[str, tuple[Any, Any]] = {}   # {model_name: (model, preprocess)}
_whisper_state: dict[str, Any] = {}            # {model_size: WhisperModel}


def _get_clip(model_name: str) -> tuple[Any, Any]:
    if model_name not in _clip_state:
        try:
            import open_clip  # type: ignore[import]
        except ImportError:
            raise EmbeddingProviderError(
                "open-clip-torch is not installed. Install with: uv sync --extra ml"
            )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        model.eval()
        _clip_state[model_name] = (model, preprocess)
    return _clip_state[model_name]


def _get_whisper(model_size: str) -> Any:
    if model_size not in _whisper_state:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError:
            raise EmbeddingProviderError(
                "faster-whisper is not installed. Install with: uv sync --extra ml"
            )
        _whisper_state[model_size] = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    return _whisper_state[model_size]


# ── Public interface ───────────────────────────────────────────────────────────


async def encode_images_clip(
    image_bytes_list: list[bytes],
    *,
    model_name: str = "ViT-B-32",
) -> list[list[float]]:
    """Return L2-normalised CLIP image embeddings.

    Args:
        image_bytes_list: Raw image bytes (JPEG, PNG, WebP, …) for each image.
        model_name:       open_clip model identifier; must match the index
                          dimension (ViT-B-32 → 512-d, ViT-L-14 → 768-d, …).

    Returns:
        List of float vectors (L2-normalised), one per input image.

    Raises:
        EmbeddingProviderError: If open-clip-torch is not installed.
    """

    def _encode() -> list[list[float]]:
        # Call _get_clip first so the open_clip availability check fires before
        # any other ML import (torch, Pillow) is attempted.
        clip_model, preprocess = _get_clip(model_name)
        import torch  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        tensors = [
            preprocess(Image.open(io.BytesIO(b))).unsqueeze(0)
            for b in image_bytes_list
        ]
        batch = torch.cat(tensors, dim=0)
        with torch.no_grad():
            features = clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().tolist()

    return await asyncio.to_thread(_encode)


async def transcribe_audio_whisper(
    audio_bytes: bytes,
    *,
    model_size: str = "base",
    language: str | None = None,
) -> str:
    """Transcribe audio bytes to text using faster-whisper.

    Args:
        audio_bytes: Raw audio content (WAV, MP3, FLAC, OGG, …).
        model_size:  Whisper variant: "tiny", "base", "small", "medium", "large-v3".
        language:    BCP-47 language code hint; None = auto-detect.

    Returns:
        Full transcript as a single whitespace-joined string.

    Raises:
        EmbeddingProviderError: If faster-whisper is not installed.
    """

    def _transcribe() -> str:
        whisper_model = _get_whisper(model_size)
        buffer = io.BytesIO(audio_bytes)
        segments, _ = whisper_model.transcribe(
            buffer, beam_size=1, language=language
        )
        return " ".join(seg.text.strip() for seg in segments)

    return await asyncio.to_thread(_transcribe)
