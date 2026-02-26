# Observability

Retrieval-OS emits three telemetry signals: **structured logs** (stdout JSON), **Prometheus metrics** (`/metrics`), and **distributed traces** (OTLP → Jaeger). All three are wired from day one and active for every request.

---

## Logs

Logs are emitted as JSON to stdout using `structlog`. Every log event includes:

```json
{
  "event": "startup.complete",
  "logger": "retrieval_os.api.main",
  "level": "info",
  "timestamp": "2026-02-25T12:00:00.123456Z",
  "environment": "production",
  "version": "0.1.0",
  "postgres": true,
  "redis": true
}
```

Log levels: `DEBUG` → `INFO` → `WARNING` → `ERROR` → `CRITICAL`. Controlled by `LOG_LEVEL` env var.

**Notable log events:**

| Event | Level | When |
|---|---|---|
| `startup.complete` | INFO | Application ready |
| `startup.dependency_check_failed` | ERROR | Postgres or Redis unreachable at startup |
| `shutdown.started` / `shutdown.complete` | INFO | Graceful shutdown |
| `background.rollout_stepper.stepped` | INFO | Deployments advanced |
| `background.rollback_watchdog.triggered` | INFO | Automatic rollback fired |
| `deployment.activated` | INFO | Rollout complete, ACTIVE |
| `deployment.rolled_back` | INFO | Manual rollback executed |
| `cache.get_error` / `cache.set_error` | WARNING | Redis error (non-fatal) |
| `query_router.redis_warm_failed` | WARNING | Redis write failed during cache warm |
| `usage.write_failed` | WARNING | Background usage record write failed |
| `background.*.error` | ERROR | Unexpected exception in background loop |

---

## Prometheus Metrics

Available at `GET /metrics`. Scraped every 15s by the local Prometheus.

All metrics are prefixed `retrieval_os_`.

### Serving Path

| Metric | Type | Labels | Description |
|---|---|---|---|
| `retrieval_os_retrieval_requests_total` | Counter | `plan_name` | Total retrieval executions. |
| `retrieval_os_retrieval_latency_seconds` | Histogram | `plan_name` | End-to-end executor latency (cache hit path included). Buckets: 10ms–5s. |
| `retrieval_os_cache_hits_total` | Counter | `plan_name` | Semantic query cache hits. |
| `retrieval_os_cache_misses_total` | Counter | `plan_name` | Semantic query cache misses. |
| `retrieval_os_embed_requests_total` | Counter | `provider` | Total embedding requests dispatched. |
| `retrieval_os_embed_latency_seconds` | Histogram | `provider` | Embedding latency. Buckets: 5ms–1s. |
| `retrieval_os_embed_errors_total` | Counter | `provider` | Embedding failures. |
| `retrieval_os_index_latency_seconds` | Histogram | `backend` | Index ANN query latency. Buckets: 2ms–500ms. |
| `retrieval_os_index_errors_total` | Counter | `backend` | Index query failures. |

**Key derived queries:**

```promql
# Cache hit rate for a plan
rate(retrieval_os_cache_hits_total{plan_name="my-docs"}[5m])
/
(rate(retrieval_os_cache_hits_total{plan_name="my-docs"}[5m]) + rate(retrieval_os_cache_misses_total{plan_name="my-docs"}[5m]))

# P99 query latency
histogram_quantile(0.99, sum(rate(retrieval_os_retrieval_latency_seconds_bucket[5m])) by (le, plan_name))

# Embedding error rate
rate(retrieval_os_embed_errors_total[5m]) / rate(retrieval_os_embed_requests_total[5m])
```

### Plans

| Metric | Type | Labels | Description |
|---|---|---|---|
| `retrieval_os_plans_total` | Gauge | — | Total projects (incremented on create). |
| `retrieval_os_plan_versions_total` | Gauge | `plan_name` | Total IndexConfig versions per project. |

### Deployments

| Metric | Type | Labels | Description |
|---|---|---|---|
| `retrieval_os_deployment_status` | Gauge | `deployment_id`, `plan_name`, `environment`, `status` | Set to 1 when the deployment is in the labelled status, 0 otherwise. Allows multi-deployment tracking. |
| `retrieval_os_deployment_traffic_weight` | Gauge | `deployment_id`, `plan_name`, `environment` | Current traffic weight (0.0–1.0). Updated by rollout stepper. |
| `retrieval_os_rollback_events_total` | Counter | `deployment_id`, `plan_name`, `triggered_by` | Rollback events. `triggered_by` = `api` or `watchdog`. |
| `retrieval_os_rollout_duration_seconds` | Histogram | `plan_name` | Seconds from deployment creation to ACTIVE. Buckets: 1min–24h. |

**Key queries:**

```promql
# Is any plan currently rolling out?
retrieval_os_deployment_status{status="ROLLING_OUT"} == 1

# Rollout progress for a specific deployment
retrieval_os_deployment_traffic_weight{plan_name="my-docs"}

# Rollback rate over 24h
increase(retrieval_os_rollback_events_total[24h])
```

### Broader metric families (reference; activated in later phases)

| Metric | Phase | Description |
|---|---|---|
| `retrieval_os_router_requests_total` | 4 | Queries routed with traffic split labels |
| `retrieval_os_query_duration_seconds` | 4 | Fine-grained query duration with `cache_hit`, `modality` |
| `retrieval_os_embed_duration_seconds` | 4 | Embed latency with `model`, `modality` labels |
| `retrieval_os_embed_tokens_total` | 7 | Token volume for cost tracking |
| `retrieval_os_embed_cost_usd_total` | 7 | Estimated cost per provider/model |
| `retrieval_os_index_query_duration_seconds` | 4 | Index latency with `collection` label |
| `retrieval_os_index_connections_active` | 8 | Active connections per backend |
| `retrieval_os_eval_recall_at_k` | 6 | Recall@k from latest eval run |
| `retrieval_os_eval_mrr` | 6 | MRR from latest eval run |
| `retrieval_os_eval_ndcg_at_k` | 6 | NDCG@k from latest eval run |
| `retrieval_os_eval_context_quality` | 6 | LLM-as-judge quality score (1–5) |
| `retrieval_os_eval_regression_alerts_total` | 6 | Regression alert fire count |
| `retrieval_os_eval_query_failure_rate` | 6 | Query failure fraction per eval run |
| `retrieval_os_lineage_artifacts_total` | 5 | Total artifacts by type |
| `retrieval_os_lineage_orphaned_artifacts_total` | 5 | Artifacts with no live deployment dependency |
| `retrieval_os_cost_usd_total` | 7 | Cumulative cost per plan/model |
| `retrieval_os_cache_efficiency_ratio` | 7 | Cache hit rate per plan (rolling) |
| `retrieval_os_recommendations_active` | 7 | Active optimization recommendations |

---

## Distributed Traces

Every request produces a trace in Jaeger. View at: `http://localhost:16686`

**Trace structure per query:**

```
retrieval_os.query  [root span]
  attrs:
    request_id = "018e7a2b-..."   (from X-Request-ID header or generated)
    plan_name = "my-docs"
    http.method = "POST"
    http.route = "/v1/query/{plan_name}"
    http.status_code = 200
    http.url = "http://localhost:8000/v1/query/my-docs"
  │
  ├── retrieval_os.cache.lookup [span]
  │     attrs: cache_key = "ros:qcache:a3b4...", cache_hit = false
  │
  ├── retrieval_os.embed [span]
  │     attrs: provider = "sentence_transformers", model = "BAAI/bge-m3"
  │
  ├── retrieval_os.index.query [span]
  │     attrs: backend = "qdrant", collection = "my_docs_v1", top_k = 20
  │
  └── retrieval_os.cache.write [span, on miss only]
        attrs: ttl_seconds = 3600
```

The `X-Request-ID` header is propagated throughout. If the caller provides one, it is used; otherwise a new UUIDv7 is generated. The same ID appears in logs, traces, and the response header.

**OTel instrumentation stack:**

- `opentelemetry-sdk` — tracer/span provider
- `opentelemetry-instrumentation-fastapi` — auto-instruments all HTTP routes
- `opentelemetry-exporter-otlp-proto-grpc` — exports to OTLP gRPC endpoint
- `opentelemetry-instrumentation-sqlalchemy` — instruments Postgres queries (adds DB spans)

---

## Alerting Rules

Alert rules are defined in `infra/prometheus/alert_rules/` and loaded by Prometheus.

### `serving.yaml`

```yaml
- alert: RetrievalHighErrorRate
  severity: warning
  expr: >
    rate(retrieval_os_embed_errors_total[5m]) > 0.01
  annotations:
    summary: "Embedding error rate > 1% over 5min"

- alert: RetrievalCriticalErrorRate
  severity: critical
  expr: >
    rate(retrieval_os_embed_errors_total[5m]) > 0.05
  annotations:
    summary: "Embedding error rate > 5% over 5min"

- alert: RetrievalHighLatencyP99
  severity: warning
  expr: >
    histogram_quantile(0.99,
      sum(rate(retrieval_os_retrieval_latency_seconds_bucket[5m])) by (le, plan_name)
    ) > 0.5
  annotations:
    summary: "Retrieval P99 latency > 500ms"
```

### `eval_regression.yaml`

```yaml
- alert: RecallRegressionWarning
  severity: warning
  expr: >
    retrieval_os_eval_recall_at_k
    - retrieval_os_eval_recall_at_k offset 24h
    < -0.05
  annotations:
    summary: "Recall@k dropped > 5% in 24h"

- alert: RecallRegressionCritical
  severity: critical
  expr: >
    retrieval_os_eval_recall_at_k
    - retrieval_os_eval_recall_at_k offset 24h
    < -0.10
  annotations:
    summary: "Recall@k dropped > 10% in 24h"
```

### `cost_anomaly.yaml`

```yaml
- alert: EmbeddingCostSpike
  severity: warning
  expr: >
    rate(retrieval_os_embed_cost_usd_total[1h])
    > 1.5 * avg_over_time(rate(retrieval_os_embed_cost_usd_total[1h])[7d:1h])
  annotations:
    summary: "Embedding cost > 1.5x 7-day moving average"
```

---

## Grafana Dashboards

Dashboards are provisioned from `infra/grafana/provisioning/dashboards/`. Access Grafana at: `http://localhost:3000` (admin/admin).

| Dashboard | Active after | Key panels |
|---|---|---|
| **Serving Health** | Phase 3 | Query latency P50/P95/P99, cache hit rate, embed latency, index latency, error rate |
| **Lineage Status** | Phase 5 | Artifact count by type, orphan count, DAG depth per plan |
| **Retrieval Quality** | Phase 6 | Recall@k trend, MRR, NDCG, context quality score, regression alert history |
| **Cost Intelligence** | Phase 7 | Cost per plan/day, cache efficiency ratio, token volume, active recommendations |

---

## SLO Reference

| Signal | Target | Measurement |
|---|---|---|
| Query P99 latency | < 200ms | `histogram_quantile(0.99, retrieval_os_retrieval_latency_seconds_bucket)` |
| Query error rate | < 0.1% | `rate(embed_errors_total + index_errors_total) / rate(retrieval_requests_total)` |
| Cache hit rate | > 20% (varies by workload) | `cache_hits_total / (cache_hits_total + cache_misses_total)` |
| Recall@10 regression | < 5% drop from baseline | `eval_recall_at_k{k="10"}` trend |
| Rollback detection latency | < 60s from threshold breach | `rollback_watchdog_interval_seconds = 30` |

---

## Request ID Propagation

Every request gets a unique `X-Request-ID`:

1. If the caller provides `X-Request-ID`, that value is used.
2. Otherwise, a fresh UUIDv7 is generated.

The ID is:
- Returned in the response `X-Request-ID` header
- Attached to the OTel root span as `retrieval_os.request_id`
- Available in logs via `request.state.request_id`

This enables full correlation across logs, traces, and error reports for a single request.
