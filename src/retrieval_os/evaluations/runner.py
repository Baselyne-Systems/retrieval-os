"""Eval job runner — loads a dataset, runs retrieval, computes metrics.

Dataset format: S3 JSONL (one record per line):
  {"query": "...", "relevant_ids": ["id1", "id2"],
   "relevant_scores": {"id1": 1.0, "id2": 0.5}}

  relevant_scores is optional; defaults to 1.0 for all relevant_ids.

The runner is called from the background eval_job_runner loop. It is NOT
async-safe to run multiple jobs concurrently (intentional — avoids
overwhelming embedding/index backends).
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field

from retrieval_os.evaluations.metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)

log = logging.getLogger(__name__)


# ── Dataset record ─────────────────────────────────────────────────────────────


@dataclass
class EvalRecord:
    query: str
    relevant_ids: set[str]
    relevance_scores: dict[str, float]  # id → score; 1.0 for binary


@dataclass
class EvalResults:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    total_queries: int
    failed_queries: int
    error_message: str | None = None
    # per-query retrieved IDs (for debugging, not stored in DB)
    query_results: list[tuple[list[str], set[str]]] = field(
        default_factory=list, repr=False
    )


# ── Dataset loading ────────────────────────────────────────────────────────────


def _parse_jsonl(raw: bytes) -> list[EvalRecord]:
    """Parse raw bytes (plain or gzip-compressed JSONL) into EvalRecord list."""
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, gzip.BadGzipFile):
        text = raw.decode("utf-8")

    records: list[EvalRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("eval.dataset.parse_error", extra={"line": line_no, "error": str(exc)})
            continue

        query = obj.get("query", "")
        relevant_ids: list[str] = obj.get("relevant_ids", [])
        if not query or not relevant_ids:
            log.warning(
                "eval.dataset.invalid_record",
                extra={"line": line_no, "reason": "missing query or relevant_ids"},
            )
            continue

        relevant_scores_raw: dict[str, float] = obj.get("relevant_scores", {})
        # Default to binary relevance (1.0) for any id not in relevant_scores
        relevance_scores = {
            rid: relevant_scores_raw.get(rid, 1.0) for rid in relevant_ids
        }
        records.append(
            EvalRecord(
                query=query,
                relevant_ids=set(relevant_ids),
                relevance_scores=relevance_scores,
            )
        )
    return records


async def load_eval_dataset(dataset_uri: str) -> list[EvalRecord]:
    """Download and parse the eval dataset from S3.

    URI formats supported:
      s3://bucket/path/to/dataset.jsonl
      s3://bucket/path/to/dataset.jsonl.gz
    """
    from retrieval_os.core.s3_client import download_object_bytes

    scheme_end = dataset_uri.index("://") + 3
    rest = dataset_uri[scheme_end:]
    bucket, _, key = rest.partition("/")
    raw = await download_object_bytes(key, bucket=bucket)
    return _parse_jsonl(raw)


# ── Job execution ──────────────────────────────────────────────────────────────


async def execute_eval_job(
    records: list[EvalRecord],
    *,
    plan_name: str,
    plan_version: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_normalize: bool,
    embedding_batch_size: int,
    index_backend: str,
    index_collection: str,
    distance_metric: str,
    top_k: int,
    reranker: str | None,
    rerank_top_k: int | None,
) -> EvalResults:
    """Run retrieval for every record and aggregate quality metrics.

    Retrieval errors on individual queries increment failed_queries but do NOT
    abort the job — we want partial metrics rather than a failed job whenever
    possible.
    """
    from retrieval_os.serving.executor import execute_retrieval

    per_query_results: list[tuple[list[str], set[str]]] = []
    failed = 0

    for record in records:
        try:
            chunks, _ = await execute_retrieval(
                plan_name=plan_name,
                version=plan_version,
                query=record.query,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_normalize=embedding_normalize,
                embedding_batch_size=embedding_batch_size,
                index_backend=index_backend,
                index_collection=index_collection,
                distance_metric=distance_metric,
                top_k=max(top_k, 10),  # always retrieve at least 10 for @10 metrics
                reranker=reranker,
                rerank_top_k=rerank_top_k,
                metadata_filters=None,
                cache_enabled=False,  # disable cache during eval for fresh results
                cache_ttl_seconds=0,
            )
            retrieved_ids = [c.id for c in chunks]
        except Exception:
            log.exception(
                "eval.query_failed",
                extra={"plan_name": plan_name, "query": record.query[:80]},
            )
            failed += 1
            retrieved_ids = []

        per_query_results.append((retrieved_ids, record.relevant_ids))

    if not per_query_results:
        return EvalResults(
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            ndcg_at_10=0.0,
            total_queries=len(records),
            failed_queries=failed,
            error_message="All queries failed",
        )

    # Aggregate Recall@k for each k
    def _mean_recall(k: int) -> float:
        vals = [
            compute_recall_at_k(retrieved, relevant, k)
            for retrieved, relevant in per_query_results
        ]
        return sum(vals) / len(vals)

    # Aggregate NDCG@k — need per-record relevance_scores for each result
    def _mean_ndcg(k: int) -> float:
        scores = [
            compute_ndcg_at_k(retrieved, records[i].relevance_scores, k)
            for i, (retrieved, _) in enumerate(per_query_results)
        ]
        return sum(scores) / len(scores)

    return EvalResults(
        recall_at_1=_mean_recall(1),
        recall_at_3=_mean_recall(3),
        recall_at_5=_mean_recall(5),
        recall_at_10=_mean_recall(10),
        mrr=compute_mrr(per_query_results),
        ndcg_at_5=_mean_ndcg(5),
        ndcg_at_10=_mean_ndcg(10),
        total_queries=len(records),
        failed_queries=failed,
        query_results=per_query_results,
    )
