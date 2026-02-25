"""Sparse (BM25-style) search via Qdrant named sparse vectors.

Vocabulary: the hashing trick maps each token to an integer in [0, VOCAB_SIZE)
via ``hash(token) % VOCAB_SIZE``.  No pre-built vocabulary file is needed —
the mapping is deterministic across all nodes.

Qdrant collections must have a named sparse vector called ``"sparse"`` to use
this module.  Create the collection with::

    from qdrant_client.models import SparseVectorParams, VectorParams, Distance

    client.create_collection(
        collection_name="my-col",
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

Token weights use normalised term-frequency (TF/total_tokens).  Full BM25 with
IDF would require corpus statistics stored alongside the collection; TF
normalisation gives reasonable quality without any extra infrastructure and is
a common starting point for production hybrid search.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from retrieval_os.serving.index_proxy import IndexHit

log = logging.getLogger(__name__)

VOCAB_SIZE: int = 2**16  # 65 536 sparse dimensions

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ── Tokeniser ─────────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenisation (no stop-word removal)."""
    return _TOKEN_RE.findall(text.lower())


def text_to_sparse_vector(text: str) -> dict[int, float]:
    """Convert text to a sparse TF vector using the hashing trick.

    Returns:
        ``{token_index: tf_weight}`` where ``tf_weight = count / total_tokens``.
        Returns an empty dict if the text contains no alphanumeric tokens.
    """
    tokens = tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {hash(tok) % VOCAB_SIZE: count / total for tok, count in counts.items()}


# ── Qdrant sparse vector search ───────────────────────────────────────────────


async def sparse_vector_search(
    *,
    collection: str,
    query: str,
    top_k: int,
    sparse_vector_name: str = "sparse",
) -> list[IndexHit]:
    """Search a Qdrant collection using a named sparse vector.

    The collection must have a sparse vector field created with
    ``SparseVectorParams()``.  If the collection has no such field or Qdrant
    is unavailable, an empty list is returned so the caller can fall back to
    pure-dense retrieval without raising.

    Args:
        collection:         Qdrant collection name.
        query:              Natural-language query string.
        top_k:              Maximum results to return.
        sparse_vector_name: Name of the sparse vector in the collection (default "sparse").

    Returns:
        List of IndexHit sorted by score descending; empty on any error.
    """
    sparse_vec = text_to_sparse_vector(query)
    if not sparse_vec:
        return []

    try:
        from qdrant_client.models import NamedSparseVector, SparseVector  # type: ignore[import]
    except ImportError:
        log.warning("sparse_search.qdrant_client_missing")
        return []

    from retrieval_os.serving.index_proxy import _get_qdrant  # noqa: PLC0415

    client = _get_qdrant()
    indices = list(sparse_vec.keys())
    values = list(sparse_vec.values())

    try:
        results = await client.search(
            collection_name=collection,
            query_vector=NamedSparseVector(
                name=sparse_vector_name,
                vector=SparseVector(indices=indices, values=values),
            ),
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        log.warning(
            "sparse_search.qdrant_error",
            extra={"collection": collection},
            exc_info=True,
        )
        return []

    return [
        IndexHit(id=str(r.id), score=r.score, payload=r.payload or {})
        for r in results
    ]
