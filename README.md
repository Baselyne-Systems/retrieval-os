# Retrieval-OS: Multimodal Retrieval-Native Inference Runtime

A production-grade serving layer that sits above vector DBs and model APIs, making multimodal retrieval-based systems coherent, observable, upgradeable, experimentable, and safe to evolve.

---

## What It Does

Orchestrates embeddings, index versions, retrieval plans, rerankers, context packing, LLM calls, traffic splits, and evaluation — without teams writing fragile glue code.

Think: CI/CD for retrieval. Deployment control for embedding/index changes. Observability for recall + ranking. Runtime intelligence for cost + latency.

---

## Stack

```
PostgreSQL 16   Redis 7.2   Qdrant   MinIO/S3
Prometheus      Grafana     Jaeger   OTel Collector
Python 3.12 + FastAPI + asyncio (no Kafka, no Celery)
```

**Why this stack:**
- **PostgreSQL** — all mutable state (plans, deployments, lineage DAG, eval results, cost ledger). JSONB for semi-structured plan configs. Recursive CTEs for lineage traversal. Row-level locking for deployment state transitions.
- **Redis** — semantic query cache (SHA-256 keyed), deployment weight reads on the serving path (refreshed async from Postgres), distributed locks (Redlock) for concurrent rollout prevention, rate limiting.
- **Qdrant** — primary vector index. gRPC, sparse+dense hybrid search, quantization API, self-hostable.
- **asyncio background loops** — all background work (rollback watchdog, gradual rollout, eval job runner, cost aggregation) runs as `asyncio` tasks in the FastAPI lifespan. No separate broker or worker process needed.
- **MinIO/S3** — artifact storage for embedding model weights, index snapshots, eval ground truth, dataset snapshots. boto3 interface; swap MinIO→S3/GCS with no code change.

**Key design decisions:**
- **UUIDv7 PKs everywhere** — time-ordered, disk-clustered, carry creation timestamp, no enumeration attack surface.
- **Serving path never reads Postgres** — deployment weights live in Redis (async-refreshed every 5s). Postgres degradation cannot cause query failures.
- **Plan validation at write time** — every `PlanVersion` in the DB is contractually valid. No defensive checks on the hot path.
- **RRF for cross-modal fusion** — rank-based, score-distribution agnostic. No calibration across modalities needed.
- **Postgres as job queue** — `eval_jobs` table with status. Background asyncio loop polls for `QUEUED` jobs. State survives process restarts; job gets picked up on next boot.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)                       │
│  /query  /plans  /deployments  /evaluations  /lineage       │
│  /intelligence  /health  /ready  /metrics                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────────────┐
        │ SERVING PATH                         │ MANAGEMENT PATH
        │ (P99 target < 200ms)                 │ (async, correctness-first)
        ▼                                      ▼
   Query Router                    Plan Manager
     ↓ (hash-based traffic split)  Deployment Controller
   Retrieval Executor              Lineage Tracker
     ↓                             Eval Engine
   Embed Router                    Cost Intelligence
     ↓
   Index Proxy (Qdrant / pgvector)

──────────────── SHARED INFRASTRUCTURE ─────────────────────
PostgreSQL 16  Redis 7.2  Qdrant  MinIO
OTel Collector  Prometheus  Grafana  Jaeger

──────────────── BACKGROUND TASKS (asyncio in-process) ──────
rollback_watchdog()     — polls every 30s, triggers rollback on threshold breach
rollout_stepper()       — increments traffic weight on gradual deployments
eval_job_runner()       — drains QUEUED eval_jobs table
cost_aggregator()       — hourly UsageRecord → CostEntry aggregation
```

### Component Boundaries

| Component | Owns | Does NOT Own | Exposes |
|---|---|---|---|
| Query Router | Traffic split state, shadow fanout | Plan content, index internals | Split weights, shadow fanout count |
| Retrieval Executor | Query orchestration, cache hit logic | Embeddings, index connections | Latency histograms, cache hit rate |
| Embed Router | Model dispatch, micro-batching | Plan config, index config | Token count, embed latency, cost per query |
| Index Proxy | Connector pool, retry logic | Embedding logic, plan config | Index latency |
| Plan Manager | Plan schemas, version history | Deployment state, live traffic | Plan version counter, validation errors |
| Deployment Controller | Rollout state machine, weight table | Plan content, eval scores | Rollout progress, rollback events |
| Lineage Tracker | DAG of artifacts, snapshot refs | Runtime query path | Lineage completeness, orphan artifacts |
| Eval Engine | Eval job queue, metric computation | Serving path, index writing | Recall@k, MRR, NDCG, regression delta |
| Cost Intelligence | Usage ledger, cost models | Live serving | Cost per query, cache efficiency |

---

## Directory Structure

```
retrieval-os/
├── pyproject.toml                        # PEP 621 project metadata, uv.lock
├── .env.example                          # All env vars documented
├── docker-compose.yml                    # Full local dev stack
├── docker-compose.infra.yml              # Infra only (Postgres, Redis, Qdrant, MinIO, Prometheus, Grafana, Jaeger)
├── Makefile                              # make dev, make test, make migrate, make lint
│
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 0001_initial_schema.py
│       ├── 0002_plans.py
│       ├── 0003_deployments.py
│       ├── 0004_lineage.py
│       └── 0005_eval.py
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.api
│   ├── k8s/                              # Phase 8
│   │   ├── api-deployment.yaml
│   │   ├── hpa.yaml
│   │   └── kustomization.yaml
│   ├── grafana/dashboards/
│   │   ├── serving-health.json
│   │   ├── retrieval-quality.json
│   │   ├── cost-intelligence.json
│   │   └── lineage-status.json
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alert_rules/
│   │       ├── serving.yaml
│   │       ├── eval_regression.yaml
│   │       └── cost_anomaly.yaml
│   └── otel/
│       └── otel-collector-config.yaml
│
└── src/
    └── retrieval_os/
        ├── core/                         # Shared kernel — no business logic
        │   ├── config.py                 # pydantic-settings root config
        │   ├── database.py               # SQLAlchemy async engine + session factory
        │   ├── redis_client.py           # Async Redis pool
        │   ├── s3_client.py              # boto3 S3 client wrapper
        │   ├── telemetry.py              # OTel tracer + meter provider setup
        │   ├── metrics.py                # Prometheus metric registry
        │   ├── exceptions.py             # Typed exception hierarchy
        │   ├── ids.py                    # UUIDv7 generator
        │   └── schemas/
        │       ├── events.py             # Internal event Pydantic schemas (no Avro needed)
        │       └── pagination.py         # Cursor pagination base
        │
        ├── plans/                        # Retrieval Plans domain
        │   ├── models.py                 # SQLAlchemy ORM: RetrievalPlan, PlanVersion
        │   ├── schemas.py                # Pydantic request/response schemas
        │   ├── repository.py             # DB access layer
        │   ├── service.py                # create, version, validate, clone
        │   ├── validators.py             # Plan config semantic validation
        │   └── router.py                 # FastAPI router
        │
        ├── deployments/                  # Deployment & Rollout domain
        │   ├── models.py                 # Deployment, TrafficWeight, RollbackEvent
        │   ├── schemas.py
        │   ├── repository.py
        │   ├── service.py                # Rollout state machine
        │   ├── traffic_splitter.py       # Weight-based routing decisions
        │   ├── rollback.py               # Automatic rollback logic + watchdog loop
        │   └── router.py
        │
        ├── serving/                      # Latency-sensitive serving path
        │   ├── query_router.py           # Traffic split + shadow fanout
        │   ├── retrieval_executor.py     # Plan dispatch, result merging
        │   ├── cache.py                  # Semantic query cache (Redis)
        │   ├── schemas.py                # QueryRequest, QueryResponse
        │   └── router.py                 # /query endpoint
        │
        ├── embeddings/                   # Embedding dispatch layer
        │   ├── base.py                   # Abstract EmbeddingProvider
        │   ├── text_provider.py          # sentence-transformers + OpenAI
        │   ├── image_provider.py         # CLIP / open-clip-torch
        │   ├── audio_provider.py         # Whisper → text → embed pipeline
        │   ├── video_provider.py         # Frame sampling + ViT mean-pool
        │   ├── cross_modal.py            # Cross-modal similarity bridge
        │   ├── batch_queue.py            # In-process micro-batching (asyncio.Queue)
        │   ├── cost_tracker.py           # Token/compute cost accounting
        │   └── registry.py              # Provider registry (name → class)
        │
        ├── indexes/                      # Index proxy layer
        │   ├── base.py                   # Abstract IndexBackend
        │   ├── qdrant_backend.py         # Qdrant gRPC client
        │   ├── pgvector_backend.py       # pgvector fallback
        │   ├── pool.py                   # Connection pool per backend
        │   └── registry.py
        │
        ├── lineage/                      # Embedding & Index Lineage domain
        │   ├── models.py                 # DatasetSnapshot, EmbeddingArtifact, IndexArtifact, LineageEdge
        │   ├── schemas.py
        │   ├── repository.py
        │   ├── dag.py                    # Recursive CTE traversal, orphan detection
        │   ├── service.py                # Register artifacts, link edges
        │   └── router.py
        │
        ├── evaluation/                   # Retrieval Evaluation Engine domain
        │   ├── models.py                 # EvalJob, EvalRun, MetricSnapshot, RegressionAlert
        │   ├── schemas.py
        │   ├── repository.py
        │   ├── service.py                # Job dispatch, result storage, alert dispatch
        │   ├── runner.py                 # asyncio eval job runner (polls QUEUED jobs)
        │   ├── metrics/
        │   │   ├── recall.py
        │   │   ├── mrr.py
        │   │   ├── ndcg.py
        │   │   ├── context_quality.py    # LLM-as-judge
        │   │   ├── latency.py            # P50/P95/P99 from usage_records
        │   │   └── cost.py
        │   ├── regression.py             # Threshold detection, alert generation
        │   ├── benchmarks.py             # BEIR / MIRACL loaders
        │   └── router.py
        │
        ├── intelligence/                 # Cost + Performance Intelligence domain
        │   ├── models.py                 # UsageRecord, CostEntry, Recommendation
        │   ├── schemas.py
        │   ├── repository.py
        │   ├── service.py                # Aggregation, trend analysis
        │   ├── aggregator.py             # asyncio periodic cost aggregation loop
        │   ├── recommender.py            # Rule-based recommendations
        │   └── router.py
        │
        └── api/                          # FastAPI application assembly
            ├── main.py                   # App factory, middleware, lifespan + background tasks
            ├── background.py             # All asyncio background loop definitions
            ├── middleware/
            │   ├── auth.py
            │   ├── rate_limit.py         # Redis sliding window
            │   ├── request_id.py         # UUIDv7 injection
            │   └── telemetry.py          # Span attachment
            └── health.py                 # /health, /ready, /metrics
```

---

## Core Data Models

### RetrievalPlan / PlanVersion

Every config change creates an immutable new version. `config_hash` is SHA-256 of the canonical config JSON — deduplicates identical configs, enables cross-plan eval sharing.

```python
class PlanVersion(Base):
    id: UUID                          # UUIDv7
    plan_id: UUID
    version: int                      # Monotonically increasing per plan
    is_current: bool

    embedding_provider: str           # "sentence_transformers" | "openai" | "clip"
    embedding_model: str              # e.g. "BAAI/bge-m3"
    embedding_dimensions: int
    modalities: list[str]             # ["text", "image", "audio", "video"]
    embedding_batch_size: int
    embedding_normalize: bool

    index_backend: str                # "qdrant" | "pgvector"
    index_collection: str
    distance_metric: str              # "cosine" | "dot" | "euclidean"
    quantization: str | None

    top_k: int                        # Candidate set size from index
    rerank_top_k: int | None          # Final set size after reranking
    reranker: str | None
    hybrid_alpha: float | None        # Dense/sparse blend (0.0–1.0)

    metadata_filters: dict | None
    cache_enabled: bool
    cache_ttl_seconds: int

    config_hash: str                  # SHA-256 of canonical config JSON
    change_comment: str
```

### Deployment

```python
class DeploymentStatus(str, Enum):
    PENDING = "pending"
    SHADOW = "shadow"           # Receives traffic, results discarded
    CANARY = "canary"           # Receives small % of live traffic
    ACTIVE = "active"           # Primary serving
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    RETIRED = "retired"

class Deployment(Base):
    id: UUID
    plan_version_id: UUID
    environment: str
    status: DeploymentStatus
    traffic_weight: float             # 0.0–1.0 live traffic
    shadow_weight: float              # 0.0–1.0 shadow (logged, results discarded)

    rollout_strategy: str             # "manual" | "gradual" | "shadow_then_canary"
    rollout_step_percent: float
    rollout_step_interval_seconds: int
    auto_rollback_enabled: bool
    rollback_on_recall_drop: float    # e.g. 0.05
    rollback_on_error_rate: float     # e.g. 0.01
    rollback_on_latency_p99_ms: int   # e.g. 500
```

### Lineage DAG

```python
class LineageArtifact(Base):
    id: UUID
    artifact_type: ArtifactType       # DATASET_SNAPSHOT | EMBEDDING_ARTIFACT | INDEX_ARTIFACT
    name: str
    version: str
    storage_uri: str                  # s3://bucket/path or qdrant://collection
    content_hash: str | None          # SHA-256 of artifact bytes
    metadata: dict                    # JSONB — type-specific fields

class LineageEdge(Base):
    parent_artifact_id: UUID
    child_artifact_id: UUID
    relationship: str                 # "produced_from" | "derived_from" | "deployed_as"
```

Traversal uses a PostgreSQL recursive CTE:
```sql
WITH RECURSIVE ancestors AS (
  SELECT parent_artifact_id, 1 AS depth
  FROM lineage_edges WHERE child_artifact_id = :start_id
  UNION ALL
  SELECT e.parent_artifact_id, a.depth + 1
  FROM lineage_edges e
  JOIN ancestors a ON e.child_artifact_id = a.parent_artifact_id
  WHERE a.depth < 20
) SELECT * FROM ancestors;
```

### EvalRun

```python
class EvalRun(Base):
    id: UUID
    eval_job_id: UUID
    plan_version_id: UUID
    completed_at: datetime
    metrics: dict       # {
                        #   "recall@5": 0.723, "recall@10": 0.841,
                        #   "mrr": 0.612, "ndcg@10": 0.589,
                        #   "latency_p50_ms": 45.2, "latency_p99_ms": 287.4,
                        #   "context_relevance_mean": 0.78,
                        #   "total_tokens_used": 485234,
                        #   "estimated_cost_usd": 0.0194,
                        #   "cache_hit_rate": 0.341
                        # }
    sample_count: int
    duration_seconds: float
```

### Background Task Pattern

```python
# src/retrieval_os/api/background.py

async def rollback_watchdog(db, redis):
    """Polls active deployments every 30s. Triggers rollback if thresholds exceeded."""
    while True:
        await asyncio.sleep(30)
        await check_and_rollback_if_needed(db, redis)

async def rollout_stepper(db, redis):
    """Advances gradual deployments one step at their configured interval."""
    while True:
        await asyncio.sleep(10)
        await advance_pending_rollout_steps(db, redis)

async def eval_job_runner(db, s3):
    """Drains QUEUED eval_jobs. One job at a time; survives restarts via DB state."""
    while True:
        job = await db.fetch_queued_eval_job()
        if job:
            await run_eval_job(job, db, s3)
        else:
            await asyncio.sleep(5)

async def cost_aggregator(db):
    """Aggregates UsageRecords into CostEntries once per hour."""
    while True:
        await asyncio.sleep(3600)
        await aggregate_usage_to_cost_entries(db)

# Wired in FastAPI lifespan:
@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(rollback_watchdog(db, redis)),
        asyncio.create_task(rollout_stepper(db, redis)),
        asyncio.create_task(eval_job_runner(db, s3)),
        asyncio.create_task(cost_aggregator(db)),
    ]
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
```

---

## API Surface

```
# Serving
POST  /v1/query                { plan_name, query: {text?, image_url?, audio_url?, video_url?}, filters?, top_k_override? }
POST  /v1/query/batch

# Plans
POST   /v1/plans
GET    /v1/plans
GET    /v1/plans/{name}
POST   /v1/plans/{name}/versions
GET    /v1/plans/{name}/versions
GET    /v1/plans/{name}/versions/{version}
POST   /v1/plans/{name}/clone
DELETE /v1/plans/{name}

# Deployments
POST   /v1/deployments
GET    /v1/deployments
GET    /v1/deployments/{id}
PATCH  /v1/deployments/{id}/promote
PATCH  /v1/deployments/{id}/rollback
PATCH  /v1/deployments/{id}/weights
GET    /v1/deployments/{id}/events
GET    /v1/environments/{env}/active

# Evaluation
POST   /v1/evaluations/jobs
GET    /v1/evaluations/jobs
GET    /v1/evaluations/jobs/{id}
GET    /v1/evaluations/jobs/{id}/results
GET    /v1/evaluations/runs
GET    /v1/evaluations/compare?run_a=&run_b=
GET    /v1/evaluations/alerts
PATCH  /v1/evaluations/alerts/{id}/acknowledge
POST   /v1/evaluations/benchmarks/{name}/run

# Lineage
POST   /v1/lineage/artifacts
GET    /v1/lineage/artifacts
GET    /v1/lineage/artifacts/{id}
POST   /v1/lineage/edges
GET    /v1/lineage/plans/{name}/graph
GET    /v1/lineage/artifacts/{id}/ancestors
GET    /v1/lineage/artifacts/{id}/descendants
GET    /v1/lineage/orphans

# Intelligence
GET    /v1/intelligence/cost/summary
GET    /v1/intelligence/cost/trends
GET    /v1/intelligence/performance/query-distribution
GET    /v1/intelligence/performance/hit-rates
GET    /v1/intelligence/recommendations
PATCH  /v1/intelligence/recommendations/{id}/dismiss

# Infrastructure
GET    /health
GET    /ready
GET    /metrics
GET    /v1/info
```

---

## Instrumentation

### Prometheus Metric Families (prefix: `retrieval_os_`)

**Query Router**
```
router_requests_total{plan_name, environment, routing_type=["live","shadow"]}  Counter
router_traffic_weight{plan_name, deployment_id}                                Gauge
```

**Retrieval Executor**
```
query_duration_seconds{plan_name, deployment_id, cache_hit, modality}   Histogram (buckets: 10ms–5s)
query_errors_total{plan_name, deployment_id, error_type}                 Counter
cache_hits_total{plan_name, deployment_id}                               Counter
cache_misses_total{plan_name, deployment_id}                             Counter
```

**Embed Router**
```
embed_duration_seconds{provider, model, modality}   Histogram
embed_tokens_total{provider, model, modality}        Counter
embed_cost_usd_total{provider, model, modality}      Counter
embed_errors_total{provider, model, error_type}      Counter
```

**Index Proxy**
```
index_query_duration_seconds{backend, collection}   Histogram
index_query_errors_total{backend, collection}        Counter
index_connections_active{backend}                    Gauge
```

**Deployment Controller**
```
deployment_status{deployment_id, plan_name, environment, status}   Gauge (0/1)
deployment_traffic_weight{deployment_id, plan_name, environment}   Gauge
rollback_events_total{deployment_id, plan_name, triggered_by}      Counter
rollout_duration_seconds{deployment_id, plan_name}                 Histogram
```

**Eval Engine**
```
eval_job_duration_seconds{plan_name}                          Histogram
eval_recall_at_k{plan_name, deployment_id, k}                 Gauge
eval_mrr{plan_name, deployment_id}                            Gauge
eval_ndcg_at_k{plan_name, deployment_id, k}                   Gauge
eval_context_quality{plan_name, deployment_id}                Gauge
eval_regression_alerts_total{plan_name, metric_name, severity} Counter
eval_query_failure_rate{plan_name}                            Gauge
```

**Cost Intelligence**
```
cost_usd_total{plan_name, entry_type, model}   Counter
cache_efficiency_ratio{plan_name}              Gauge
recommendations_active{category, priority}     Gauge
```

### OTel Trace Structure (per query)
```
retrieval_os.query [root]
  span attrs: query_id, plan_name, plan_version, deployment_id, modality, cache_hit, result_count
  ├── retrieval_os.cache.lookup
  ├── retrieval_os.embed.{modality}
  │     └── retrieval_os.embed.provider.{name}
  ├── retrieval_os.index.query
  │     └── retrieval_os.index.{backend}.{collection}
  ├── retrieval_os.rerank  [optional]
  └── retrieval_os.cache.write  [on miss]
```

### Alerting Rules

```yaml
# Serving
- alert: RetrievalHighErrorRate
  expr: rate(retrieval_os_query_errors_total[5m]) / rate(retrieval_os_router_requests_total[5m]) > 0.01

- alert: RetrievalHighLatencyP99
  expr: histogram_quantile(0.99, retrieval_os_query_duration_seconds) > 0.5

# Eval regression
- alert: RecallRegressionWarning
  expr: retrieval_os_eval_recall_at_k - retrieval_os_eval_recall_at_k offset 24h < -0.05

- alert: RecallRegressionCritical
  expr: retrieval_os_eval_recall_at_k - retrieval_os_eval_recall_at_k offset 24h < -0.10

# Cost anomaly
- alert: EmbeddingCostSpike
  expr: rate(retrieval_os_cost_usd_total[1h]) > 1.5 * avg_over_time(rate(retrieval_os_cost_usd_total[1h])[7d:1h])
```

---

## Build Phases

### Phase 1 — Foundation (1.5 weeks)
**Goal:** Running service, full infra via docker-compose, no business logic.

**Deliverables:**
- `pyproject.toml` with all dependencies pinned (uv)
- `docker-compose.infra.yml`: Postgres 16, Redis 7.2, Qdrant 1.9, MinIO, Prometheus, Grafana, Jaeger
- `docker-compose.yml`: adds API service
- `core/` fully implemented: config, database engine, Redis pool, S3 client, OTel setup, Prometheus registry, exception hierarchy, UUIDv7 generator
- `api/main.py`: FastAPI app factory with lifespan hooks (DB pool warm-up, Redis ping), middleware stack
- `/health`, `/ready`, `/metrics` endpoints live
- `api/background.py`: stub background task loop wired into lifespan (loops exist but do nothing yet)
- Alembic initialized with empty migration
- `Makefile`: `make dev`, `make test`, `make lint`, `make migrate`

**Metrics unlocked:**
- `up` (scrape target health)
- `process_cpu_seconds_total`, `process_resident_memory_bytes`
- `/ready` latency (infrastructure connectivity canary)

**Success gate:**
- `docker-compose up` → `GET /ready` returns `{"postgres":"ok","redis":"ok","qdrant":"ok"}` within 30s
- `GET /metrics` returns valid Prometheus text
- Structured JSON logs visible in container output

---

### Phase 2 — Retrieval Plans (2 weeks)
**Goal:** Plans are immutable, versioned, and semantically validated. Everything else foreign-keys against them.

**Deliverables:**
- `alembic/versions/0002_plans.py`
- Full `plans/` domain: models, schemas (Pydantic v2), repository, service, validators, router
- Validators: embedding model in provider registry; index backend in backend registry; `rerank_top_k <= top_k`; modalities non-empty; distance metric compatible with embedding model
- `config_hash`: `SHA-256(canonical_json(embedding_provider, model, index_backend, collection, distance_metric, top_k, rerank_top_k, hybrid_alpha, metadata_filters))`
- Version creation: increments version integer, sets `is_current=True` on new, `is_current=False` on previous — inside a single transaction
- Duplicate `config_hash` within same plan returns HTTP 409
- All plan CRUD + clone endpoints

**Metrics unlocked:**
- `retrieval_os_plans_total`, `retrieval_os_plan_versions_total`
- API request durations via middleware

**Success gate:**
- Invalid embedding model → HTTP 422 with field-level error
- Duplicate config hash → HTTP 409
- 20 concurrent version creates → unique monotonically increasing versions (asyncio.gather test)
- List endpoint returns cursor-paginated results with correct total

---

### Phase 3 — Serving Path (3 weeks)
**Goal:** End-to-end query execution. First contact with all external dependencies. Must meet latency targets from day one.

**Deliverables:**
- `embeddings/`: text (sentence-transformers + OpenAI), image (CLIP/open-clip), audio (Whisper→text→embed), video (frame sample + ViT mean-pool), cross_modal bridge, micro-batch queue (`asyncio.Queue` + 10ms flush window), cost tracker, provider registry
- `indexes/`: Qdrant gRPC backend (connection pool, tenacity retry), pgvector fallback, pool with health checking, backend registry
- `serving/`:
  - `cache.py`: key = `SHA-256(plan_version_id + base64(quantized_4dp_vector))`, stored in Redis HASH with plan's `cache_ttl_seconds`
  - `retrieval_executor.py`: orchestrates embed → cache lookup → index query → optional rerank → cache write; full OTel spans + Prometheus histograms on every step
  - `query_router.py`: reads deployment weights from Redis (5s refresh), routes via `hash(query_id) % 100 < weight * 100`, fires shadow queries as `asyncio.create_task()` (fire-and-forget, errors logged)
- After each query: `asyncio.create_task(write_usage_record(query_result, db))` — non-blocking write to `usage_records` table

**Metrics unlocked:**
- `retrieval_os_query_duration_seconds` — **primary SLO metric**
- `retrieval_os_cache_hits_total` / `retrieval_os_cache_misses_total`
- `retrieval_os_embed_tokens_total`, `retrieval_os_embed_cost_usd_total`
- `retrieval_os_index_query_duration_seconds`
- Full distributed traces in Jaeger per query
- **"Serving Health" Grafana dashboard functional**

**Success gate:**
- P99 < 200ms text queries at 50 RPS for 2 min (Locust)
- Shadow fanout doesn't increase primary path P99 (compare shadow_weight=0 vs 1.0)
- Cache hit rate > 30% with 40% repeated queries in test dataset
- Zero unhandled exceptions during load test

---

### Phase 4 — Deployment & Rollouts (2 weeks)
**Goal:** Changing a plan in production is a controlled, measurable, reversible operation.

**Deliverables:**
- `alembic/versions/0003_deployments.py`
- Full `deployments/` domain: models, schemas, repository, service, traffic_splitter, rollback, router
- State machine with pre-condition gates:
  - `PENDING → SHADOW`: index collection must exist and respond
  - `SHADOW → CANARY`: min shadow query count (default 100) + no active regression alerts
  - `CANARY → ACTIVE`: completed eval run with no regressions, or manual override with reason
  - Any → `ROLLING_BACK`: auto-watchdog or manual API
  - `ROLLING_BACK → ROLLED_BACK`: previous deployment's weight restored
- `traffic_splitter.py`: `hash(query_id) % 100 < weight * 100` — consistent, stateless
- `rollback_watchdog()` background loop (now activated, not a stub): queries Prometheus every 30s, reads `rollback_on_error_rate` / `rollback_on_latency_p99_ms` from deployment config, triggers rollback via service layer
- `rollout_stepper()` background loop activated: increments `traffic_weight` by `rollout_step_percent` every `rollout_step_interval_seconds` for `gradual` strategy deployments
- Redlock (Redis) prevents concurrent state transitions on the same deployment
- Full deployment API endpoints

**Metrics unlocked:**
- `retrieval_os_deployment_status` gauge — enables multi-deployment comparison in Grafana
- `retrieval_os_deployment_traffic_weight` — visible rollout progress
- `retrieval_os_rollback_events_total{triggered_by}` — operational safety signal
- `retrieval_os_rollout_duration_seconds`

**Success gate:**
- Auto-rollback fires within 60s of injecting 5% error rate (end-to-end test with mock error injection)
- Two simultaneous promotion requests → second returns HTTP 409
- Zero traffic to PENDING or ROLLED_BACK deployments (verified with query log analysis)

---

### Phase 5 — Embedding & Index Lineage (1.5 weeks)
**Goal:** "What dataset is this index built from?" answerable in a single API call. Orphaned artifacts are visible.

**Deliverables:**
- `alembic/versions/0004_lineage.py`
- Full `lineage/` domain: models, schemas, repository, DAG (recursive CTE), service, router
- Orphan detection: artifacts with no outbound edge to an `INDEX_ARTIFACT` referenced by any current `PlanVersionArtifact`
- S3 metadata verification on artifact registration (fetches `Content-Length` + `ETag`, stored in `metadata`)
- Cycle prevention: ancestor query run before any edge insert; cycle → HTTP 400
- `lineage_status` Grafana dashboard provisioned

**Metrics unlocked:**
- `retrieval_os_lineage_artifacts_total{artifact_type}`
- `retrieval_os_lineage_orphaned_artifacts_total` — data hygiene signal
- `retrieval_os_lineage_dag_depth{plan_name}`

**Success gate:**
- Ancestor query < 50ms for DAGs up to 100 nodes
- S3 URI that doesn't exist → HTTP 404 with `ARTIFACT_STORAGE_NOT_FOUND`
- Cycle attempt → HTTP 400

---

### Phase 6 — Retrieval Evaluation Engine (3 weeks)
**Goal:** Every plan version has a numeric quality score. Regressions are caught before production.

**Deliverables:**
- `alembic/versions/0005_eval.py`
- Full `evaluation/` domain: models, schemas, repository, service, metrics subpackage, regression, benchmarks, router
- `runner.py`: `eval_job_runner()` asyncio loop (already wired in lifespan from Phase 1 stub) — polls `eval_jobs` for `QUEUED`, sets `RUNNING`, executes, writes `EvalRun`, detects regressions, sets `COMPLETED`/`FAILED`
- `metrics/recall.py`: `recall@k = |retrieved[:k] ∩ relevant| / |relevant|`, averaged over queries
- `metrics/mrr.py`: `1 / rank_of_first_relevant`, averaged; 0.0 if none relevant in results
- `metrics/ndcg.py`: `sklearn.metrics.ndcg_score` with binary or graded relevance
- `metrics/context_quality.py`: LLM-as-judge on 10% sample; structured prompt → score 1–5; rate-limited + cost-tracked
- `metrics/latency.py`: P50/P95/P99 from `usage_records` table for the eval window
- `regression.py`: compare new `EvalRun` against previous version's run; fire `RegressionAlert` on threshold breach (default: 5% relative drop on recall@10, 10ms absolute on P99)
- Ground truth JSONL: `{"query_id": "q001", "query": {"text": "..."}, "relevant_ids": [...], "relevance_scores": {...}}`
- BEIR + MIRACL benchmark loaders
- Alertmanager push integration on `RegressionAlert` creation
- **"Retrieval Quality" Grafana dashboard provisioned**

**Metrics unlocked:**
- `retrieval_os_eval_recall_at_k{k=1,3,5,10}` per plan version
- `retrieval_os_eval_mrr`, `retrieval_os_eval_ndcg_at_k`
- `retrieval_os_eval_context_quality`
- `retrieval_os_eval_regression_alerts_total` — **operational quality health signal**
- `retrieval_os_eval_query_failure_rate`

**Success gate:**
- 1,000-query eval job completes in < 5 min
- Recall@10 matches hand-calculated expected within 0.001
- Regression alert fires within 2 min of eval completion
- LLM judge respects ≤ 10 calls/min (mock judge counts calls)

---

### Phase 7 — Cost + Performance Intelligence (2 weeks)
**Goal:** Operators see total cost per plan/day and receive specific optimization suggestions.

**Deliverables:**
- Full `intelligence/` domain: models, schemas, repository, service, aggregator, recommender, router
- `aggregator.py`: `cost_aggregator()` asyncio loop (already wired from Phase 1 stub) — runs hourly, aggregates `usage_records` into `cost_entries` using `model_pricing` table (per-token costs with effective date ranges)
- Rule-based recommender (each rule: `evaluate(plan_version, metrics_window) → Recommendation | None`):
  - `LowCacheEfficiencyRule`: cache_hit_rate < 10%, volume > 1k/day → suggest increase cache TTL
  - `HighEmbedCostRule`: embed cost > 80% of total → suggest cheaper model + eval quality delta
  - `OversizedTopKRule`: top_k >> rerank_top_k → suggest reducing top_k to `rerank_top_k * 3`
  - `IndexRebuildFrequencyRule`: 3+ rebuilds in 7 days, no eval improvement → investigate dataset churn
  - `UnusedShadowDeploymentRule`: SHADOW state > 7 days → review or retire
- Recommendations persist with 30-day expiry, deduplicated by `(plan_version_id, rule_name)`
- **"Cost Intelligence" Grafana dashboard provisioned**

**Metrics unlocked:**
- `retrieval_os_cost_usd_total{plan_name,entry_type,model}`
- `retrieval_os_cache_efficiency_ratio{plan_name}` — key ROI signal
- `retrieval_os_recommendations_active{category,priority}`

**Success gate:**
- Cost summary within 5% of manually summed `usage_records`
- Recommendations generated within 5 min of conditions being met
- No duplicate recommendations under concurrent evaluation (idempotency test)

---

### Phase 8 — Production Hardening + Advanced Multimodal (3 weeks)
**Goal:** Kubernetes-deployable, chaos-tested, full cross-modal retrieval.

**Deliverables:**
- **K8s**: `api-deployment.yaml` (3 replicas min), HPA on P99 latency via Prometheus adapter, graceful SIGTERM (drain in-flight requests, cancel background tasks, close pools)
- **Circuit breaker** in `indexes/pool.py` and embedding providers: 3 failures in 10s → open for 30s → probe
- **Idempotent query endpoint**: provided `query_id` + result in cache → return immediately (24h TTL idempotency cache, separate from semantic cache)
- **Cross-modal RRF**: text query against `modalities: ["image","text"]` → CLIP text embed + text embed → parallel index queries → Reciprocal Rank Fusion: `score(d) = Σ 1/(60 + rank_i(d))`
- **Video queries**: sample 1 frame/sec (max 30), CLIP embed each frame, mean-pool → image index
- **Audio queries**: Whisper transcription → text embed → text index (+ optional cross-modal expansion)
- **Security**: API key auth (bcrypt-hashed), payload size limits at middleware (10MB image URL, 100MB audio, 500MB video), parameterized query audit
- **Load tests**: `tests/load/locustfile.py` — 100 RPS sustained text, 500 RPS burst, multimodal mixed 50 RPS

**Metrics unlocked:**
- `retrieval_os_circuit_breaker_state{backend}` — 0=closed, 1=open (critical operational signal)
- `retrieval_os_rate_limit_hits_total{tenant_id,plan_name}`
- Per-modality labels on all existing histogram metrics
- `retrieval_os_cross_modal_fusion_duration_seconds`

**Success gate:**
- Kill one API pod under load → < 1% failure rate (Locust chaos test)
- Circuit breaker opens within 15s of 100% Qdrant error rate; closes within 60s of recovery
- P99 < 200ms text / < 800ms cross-modal at 100 RPS for 10 min
- Same `query_id` twice → second call returns `cache_hit: true`

---

## Metric Availability Timeline

| Metric | Phase |
|---|---|
| Service liveness/readiness | 1 |
| Process CPU/memory | 1 |
| Plan count, version count | 2 |
| **Query latency P50/P95/P99** | 3 |
| Cache hit rate | 3 |
| Embedding cost per query | 3 |
| Full distributed traces | 3 |
| **Deployment traffic weight + rollout progress** | 4 |
| Rollback event count | 4 |
| Lineage orphan count | 5 |
| **Recall@k, MRR, NDCG** | 6 |
| Context quality score | 6 |
| **Regression alert count** | 6 |
| Total cost by plan/modality | 7 |
| Cache efficiency ratio | 7 |
| Active recommendations | 7 |
| Circuit breaker state | 8 |
| Per-modality latency | 8 |

## 4 Grafana Dashboards

| Dashboard | Functional After |
|---|---|
| Serving Health | Phase 3 |
| Lineage Status | Phase 5 |
| Retrieval Quality | Phase 6 |
| Cost Intelligence | Phase 7 |
