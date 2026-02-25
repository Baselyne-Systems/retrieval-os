"""Word-boundary text chunker with configurable size and overlap.

Uses simple whitespace tokenisation — suitable for general prose and
technical documentation. For code, structured data, or Asian scripts
swap out the tokeniser but keep the same interface.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split *text* into overlapping word-token chunks.

    Args:
        text:       Input text to split.
        chunk_size: Maximum number of words per chunk.
        overlap:    Number of words from the end of the previous chunk
                    that are prepended to the next chunk.  Must be
                    strictly less than *chunk_size*.

    Returns:
        List of non-empty string chunks.  Returns ``[]`` for blank input.

    Raises:
        ValueError: If ``overlap >= chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    words = _WORD_RE.findall(text)
    if not words:
        return []

    # Short documents that fit in a single chunk
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        chunks.append(" ".join(window))
        if i + chunk_size >= len(words):
            break
        i += step

    return chunks


def estimate_chunk_count(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> int:
    """Fast word-count-based estimate of the number of chunks.

    Does not actually perform the split — useful for quota checks before
    processing large documents.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )
    word_count = len(_WORD_RE.findall(text))
    if word_count == 0:
        return 0
    if word_count <= chunk_size:
        return 1
    step = chunk_size - overlap
    return max(1, -(-(word_count - chunk_size) // step) + 1)
