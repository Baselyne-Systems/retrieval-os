"""Prometheus metric definitions for the entire retrieval-os system.

All metrics are declared here at import time so they are registered with the
default CollectorRegistry before the first scrape. Each domain imports only the
metrics it needs.

Naming convention: retrieval_os_{component}_{metric}_{unit}
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Query Router ──────────────────────────────────────────────────────────────

router_requests_total = Counter(
    "retrieval_os_router_requests_total",
    "Total query requests routed",
    ["project_name", "environment", "routing_type"],
)

router_traffic_weight = Gauge(
    "retrieval_os_router_traffic_weight",
    "Current live traffic weight for a deployment",
    ["project_name", "deployment_id"],
)

# ── Retrieval Executor ─────────────────────────────────────────────────────────

query_duration_seconds = Histogram(
    "retrieval_os_query_duration_seconds",
    "End-to-end query latency in seconds",
    ["project_name", "deployment_id", "cache_hit", "modality"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0],
)

query_errors_total = Counter(
    "retrieval_os_query_errors_total",
    "Total query errors",
    ["project_name", "deployment_id", "error_type"],
)

cache_hits_total = Counter(
    "retrieval_os_cache_hits_total",
    "Semantic query cache hits",
    ["project_name"],
)

cache_misses_total = Counter(
    "retrieval_os_cache_misses_total",
    "Semantic query cache misses",
    ["project_name"],
)

# Phase 3 serving path metrics
retrieval_requests_total = Counter(
    "retrieval_os_retrieval_requests_total",
    "Total retrieval executions",
    ["project_name"],
)

retrieval_latency_seconds = Histogram(
    "retrieval_os_retrieval_latency_seconds",
    "End-to-end retrieval executor latency in seconds",
    ["project_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0],
)

# ── Embed Router ──────────────────────────────────────────────────────────────

embed_duration_seconds = Histogram(
    "retrieval_os_embed_duration_seconds",
    "Embedding latency in seconds",
    ["provider", "model", "modality"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

embed_tokens_total = Counter(
    "retrieval_os_embed_tokens_total",
    "Total tokens submitted for embedding",
    ["provider", "model", "modality"],
)

embed_cost_usd_total = Counter(
    "retrieval_os_embed_cost_usd_total",
    "Cumulative embedding cost in USD",
    ["provider", "model", "modality"],
)

embed_errors_total = Counter(
    "retrieval_os_embed_errors_total",
    "Embedding errors",
    ["provider"],
)

embed_requests_total = Counter(
    "retrieval_os_embed_requests_total",
    "Total embedding requests",
    ["provider"],
)

embed_latency_seconds = Histogram(
    "retrieval_os_embed_latency_seconds",
    "Embedding latency in seconds",
    ["provider"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ── Index Proxy ───────────────────────────────────────────────────────────────

index_query_duration_seconds = Histogram(
    "retrieval_os_index_query_duration_seconds",
    "Index ANN query latency in seconds",
    ["backend", "collection"],
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

index_query_errors_total = Counter(
    "retrieval_os_index_query_errors_total",
    "Index query errors",
    ["backend", "collection", "error_type"],
)

index_errors_total = Counter(
    "retrieval_os_index_errors_total",
    "Index errors (serving path)",
    ["backend"],
)

index_latency_seconds = Histogram(
    "retrieval_os_index_latency_seconds",
    "Index query latency in seconds (serving path)",
    ["backend"],
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

index_connections_active = Gauge(
    "retrieval_os_index_connections_active",
    "Active index backend connections",
    ["backend"],
)

# ── Projects ──────────────────────────────────────────────────────────────────

projects_total = Gauge(
    "retrieval_os_projects_total",
    "Total retrieval projects",
)

index_configs_total = Gauge(
    "retrieval_os_index_configs_total",
    "Total index configs",
    ["project_name"],
)

# ── Deployment Controller ─────────────────────────────────────────────────────

deployment_status = Gauge(
    "retrieval_os_deployment_status",
    "Deployment status (1 = current status label matches, 0 otherwise)",
    ["deployment_id", "project_name", "environment", "status"],
)

deployment_traffic_weight = Gauge(
    "retrieval_os_deployment_traffic_weight",
    "Live traffic fraction for this deployment (0.0–1.0)",
    ["deployment_id", "project_name", "environment"],
)

rollback_events_total = Counter(
    "retrieval_os_rollback_events_total",
    "Deployment rollback events",
    ["deployment_id", "project_name", "triggered_by"],
)

rollout_duration_seconds = Histogram(
    "retrieval_os_rollout_duration_seconds",
    "Time in seconds from deployment creation to ACTIVE status",
    ["project_name"],
    buckets=[60, 300, 600, 1800, 3600, 7200, 86400],
)

# ── Evaluation Engine ─────────────────────────────────────────────────────────

eval_job_duration_seconds = Histogram(
    "retrieval_os_eval_job_duration_seconds",
    "Evaluation job total runtime in seconds",
    ["project_name"],
    buckets=[30, 60, 120, 300, 600, 1800, 3600],
)

eval_recall_at_k = Gauge(
    "retrieval_os_eval_recall_at_k",
    "Recall@k from the latest completed eval run",
    ["project_name", "deployment_id", "k"],
)

eval_mrr = Gauge(
    "retrieval_os_eval_mrr",
    "Mean Reciprocal Rank from the latest completed eval run",
    ["project_name", "deployment_id"],
)

eval_ndcg_at_k = Gauge(
    "retrieval_os_eval_ndcg_at_k",
    "NDCG@k from the latest completed eval run",
    ["project_name", "deployment_id", "k"],
)

eval_context_quality = Gauge(
    "retrieval_os_eval_context_quality",
    "LLM-as-judge context quality score (1–5) from latest eval run",
    ["project_name", "deployment_id"],
)

eval_regression_alerts_total = Counter(
    "retrieval_os_eval_regression_alerts_total",
    "Retrieval quality regression alerts fired",
    ["project_name", "metric_name", "severity"],
)

eval_query_failure_rate = Gauge(
    "retrieval_os_eval_query_failure_rate",
    "Fraction of queries that failed in the latest eval run",
    ["project_name"],
)

# ── Lineage ───────────────────────────────────────────────────────────────────

lineage_artifacts_total = Gauge(
    "retrieval_os_lineage_artifacts_total",
    "Total registered lineage artifacts by type",
    ["artifact_type"],
)

lineage_orphaned_artifacts_total = Gauge(
    "retrieval_os_lineage_orphaned_artifacts_total",
    "Lineage artifacts with no active deployment dependency",
    ["artifact_type"],
)

lineage_dag_depth = Gauge(
    "retrieval_os_lineage_dag_depth",
    "Maximum DAG depth for project lineage",
    ["project_name"],
)

# ── Cost Intelligence ─────────────────────────────────────────────────────────

cost_usd_total = Counter(
    "retrieval_os_cost_usd_total",
    "Cumulative estimated cost in USD",
    ["project_name", "entry_type", "model"],
)

cache_efficiency_ratio = Gauge(
    "retrieval_os_cache_efficiency_ratio",
    "Cache hit rate (cache_hits / total_queries) for a project",
    ["project_name"],
)

recommendations_active = Gauge(
    "retrieval_os_recommendations_active",
    "Number of active cost/performance recommendations",
    ["category", "priority"],
)
