# Architecture

Retrieval-OS is a production control plane for RAG and semantic search systems. It sits between your application and your vector database, and provides versioning, deployment control, quality measurement, and automatic rollback for retrieval pipelines — the operational controls that vector databases and embedding APIs do not provide on their own.

---

## Core Concepts

Before reading the technical design, it helps to understand what each concept maps to from a user's perspective.

### Plan

A **Plan** is the complete specification for a retrieval pipeline: which embedding model encodes queries and documents, how documents are chunked before indexing, which vector collection they go into, how many results to retrieve, whether to rerank, and cache settings.

When you want to change any of these — swap the model, tighten chunking, add a reranker — you create a new **version** of the Plan. The previous version is preserved and still queryable. Versions are numbered sequentially and are immutable once created. Think of them as commits to your retrieval configuration.

### Deployment

A **Deployment** activates a plan version for live traffic. You can deploy instantly (100% traffic switch) or gradually (start at a low traffic weight, advance automatically on a schedule, promote to full traffic when stable). At most one deployment is live per plan at any time.

Deployments are the gate between "a config I wrote" and "a config serving real queries". The deployment record captures when a version went live, what traffic share it held, and — if it was rolled back — why.

### Ingestion Job

An **Ingestion Job** loads documents into the vector index using a specific plan version's settings. You push documents via API; the system chunks them, embeds them, and upserts the vectors into the correct collection. The ingestion job records which plan version processed each document, so the index always has a clear provenance chain.

### Evaluation Job

An **Evaluation Job** measures retrieval quality for a deployed plan version. You provide labelled query–document pairs (ground truth); the system runs each query through the live retrieval pipeline and computes Recall@k, MRR, and NDCG. Results are compared to the previous deployment's baseline. A drop beyond a configurable threshold fires a regression alert.

### Deployment Guard-Rails

When creating a deployment, you can attach quality thresholds: a minimum Recall@5, a maximum error rate, or both. The rollback watchdog checks active deployments against the latest completed evaluation. If a threshold is breached, the deployment is rolled back automatically — no human required.

### Lineage

Every artifact produced by the system — a chunked dataset, an embedding run, an index snapshot — is recorded as a node in a lineage graph, with directed edges linking parents to children. You can trace any document chunk in the index back to its source file, the embedding model that encoded it, and the plan version that owns it.

---

## The Core Problem

Retrieval-augmented systems break in silent ways. You swap an embedding model, rebuild your index with slightly different settings, or adjust top-k — and recall drops 12% a week later with no clear cause. There is no equivalent of "git diff" for retrieval config. No controlled rollout. No automatic quality regression detection.

Retrieval-OS solves this by:

1. Making every retrieval config an immutable, versioned object (a **Plan**)
2. Treating index changes as **Deployments** with traffic weights and rollback capability
3. Measuring **Recall@k, MRR, NDCG** continuously against a held-out ground truth
4. Tracking the full **lineage** from training dataset to deployed index
5. Aggregating **cost and latency** so infrastructure decisions are evidence-based

---

## Three Paths Through the System

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                 API Gateway (FastAPI)                                 │
│   /v1/query/{plan}   /v1/plans/{name}/ingest   /v1/plans   /v1/plans/{n}/deployments  │
│   /health            /ready                    /metrics    /v1/info                   │
└────────┬─────────────────────┬──────────────────────────┬────────────────────────────┘
         │                     │                          │
  ╔══════▼══════╗       ╔══════▼══════╗           ╔══════▼══════╗
  ║ SERVING PATH ║       ║  INGESTION  ║           ║ MANAGEMENT  ║
  ║  P99 < 200ms ║       ║    PATH     ║           ║    PATH     ║
  ╚══════╤══════╝       ╚══════╤══════╝           ╚══════╤══════╝
         │                     │                          │
  Query Router          Creates QUEUED            Plan Manager
  (Redis config read)   IngestionJob              (Postgres read/write)
         │                     │
  Retrieval Executor    Background runner          Deployment Controller
  (cache → embed → ANN) picks up job              (state machine + Redis)
         │                     │
  Embed Router          Chunk documents            Lineage Tracker
  (ST / OpenAI)         Embed in batches           (DAG + orphan detect)
         │                     │
  Index Proxy           Upsert → Qdrant            Eval Engine
  (Qdrant gRPC)         Register lineage           (Recall@k, MRR, NDCG)
         │                     │
  Redis Cache           Fire webhook               Cost Intelligence
  (SHA-256 keyed)       Mark COMPLETED             (usage aggregation)
```

---

## Serving Path

The hot path processes a query end-to-end. P99 target: 200ms.

```
POST /v1/query/{plan_name}
        │
        ▼
 Redis GET ros:plan:{name}:current     ← cache miss → Postgres read, then cache
        │
        ▼
 Redis GET ros:qcache:{sha256}         ← cache hit → return immediately
        │
        ▼
 embed_text(query)                     ← sentence-transformers or OpenAI
        │
        ▼
 Qdrant ANN search (gRPC)              ← top-k nearest neighbours
        │
        ▼
 (optional) rerank                     ← cross-encoder or Cohere
        │
        ▼
 Redis SET ros:qcache:{sha256}
        │
        ▼
 Return results + fire-and-forget usage_record write
```

### Serving path invariants

- **No Postgres reads on the hot path.** Plan config lives in Redis (`ros:plan:{name}:current`, 30s TTL). A Postgres failure cannot cause query failures. On Redis miss, one Postgres read warms the cache.
- **Plan validation at write time.** Every `PlanVersion` row in Postgres is guaranteed valid by the service layer. The serving path performs zero defensive checks.
- **Usage records are non-blocking.** After every query, a fire-and-forget `asyncio.create_task()` writes a `usage_record` row. The response is returned before this completes.

---

## Ingestion Path

Documents are loaded into the vector index asynchronously via a background job runner.

```
POST /v1/plans/{name}/ingest
        │
        ▼
 Validate plan version exists (Postgres)
        │
        ▼
 Create IngestionJob (status: QUEUED)
        │
        │  ← HTTP 202 returned here — job runs in background
        │
        │  ingestion_job_runner polls every 5s
        │  SELECT FOR UPDATE SKIP LOCKED on ingestion_jobs WHERE status='QUEUED'
        ▼
 1. Load plan version config from Postgres
        │
        ▼
 2. Load documents
    ├── inline: JSON array in request body
    └── S3:     download JSONL from s3://bucket/key
        │
        ▼
 3. Chunk each document (word-boundary)
    chunk_size and overlap come from the request (not the plan version)
    each chunk carries: doc_id, chunk_idx, plan_name, plan_version, doc metadata
        │
        ▼
 4. For each batch of 50 chunks:
    ├── embed_text → vectors  (uses plan version's provider + model)
    │         │
    │         └── [on first batch] ensure_collection in Qdrant
    │                              (creates collection if absent, using actual dimension)
    ├── upsert_vectors → Qdrant
    └── [on embed/upsert failure] failed_chunks += batch_size, continue next batch
        │
        ▼
 5. Register lineage DAG (idempotent)
    DatasetSnapshot ──► EmbeddingArtifact ──► IndexArtifact
        │
        ▼
 6. Fire webhook (plan.version_created) if indexed_chunks > 0
        │
        ▼
 7. Mark job COMPLETED (indexed_chunks, failed_chunks, total_chunks recorded)
    [unrecoverable error at any step → FAILED with error_message]
```

### Ingestion path properties

- **Per-batch error isolation.** An embed or upsert failure on one batch does not abort the job. That batch's chunks are counted as `failed_chunks`; the remaining batches continue. The job reaches `COMPLETED` even if some chunks failed — allowing partial indexing.
- **Collection auto-creation.** `ensure_collection` is called exactly once per job (before the first upsert), using the actual vector dimension inferred from the first embed result. If the collection already exists it is a no-op.
- **Chunk payloads are queryable.** Every vector in Qdrant carries a payload with `doc_id`, `chunk_idx`, `plan_name`, `plan_version`, and all metadata from the source document. These are filterable at query time.
- **Lineage is idempotent.** Re-running a job for the same plan version does not create duplicate lineage artifacts. Each artifact is keyed by its storage URI and skipped if it already exists.
- **Horizontal scaling.** `SELECT FOR UPDATE SKIP LOCKED` means multiple API replicas can each run an ingestion runner without processing the same job twice.

---

## Management Path

The management path handles all writes to plans, deployments, and configuration. These operations are infrequent relative to queries and have no latency SLO.

### Management path invariants

- **Version numbers are monotonic and gapless.** `SELECT FOR UPDATE` on the parent `RetrievalPlan` row serialises concurrent version creates.
- **Identical configs are rejected.** `config_hash` = SHA-256 of behavioural fields. Duplicate hash → HTTP 409 before DB write.
- **At most one live deployment per plan.** The service layer enforces this; there is no DB unique constraint to race against because the check + insert happen within a transaction with the parent plan row locked.

---

## Background Tasks

All background work runs as asyncio tasks in the FastAPI lifespan — no separate process, no message broker.

```
FastAPI lifespan
  ├── rollback_watchdog()     every 30s
  │     Checks active deployments against their guard-rail thresholds.
  │     Phase 6 populates eval metrics; watchdog reads them here.
  │
  ├── rollout_stepper()       every 10s
  │     Finds all ROLLING_OUT deployments, advances traffic_weight by
  │     rollout_step_percent. Promotes to ACTIVE when weight reaches 1.0.
  │
  ├── ingestion_job_runner()  every 5s
  │     SELECT FOR UPDATE SKIP LOCKED on ingestion_jobs WHERE status='QUEUED'.
  │     Runs the full ingestion pipeline for one job per cycle:
  │     chunk → embed → upsert → lineage → webhook → complete.
  │
  ├── eval_job_runner()       every 5s
  │     SELECT FOR UPDATE SKIP LOCKED on eval_jobs WHERE status='QUEUED'.
  │     Processes one job per cycle. State survives restarts via DB status column.
  │
  └── cost_aggregator()       every 3600s
        Groups usage_records by (plan_name, plan_version, hour).
        Upserts into cost_entries. Idempotent over restart.
```

All loops share the same pattern:

```python
while True:
    try:
        await asyncio.sleep(interval)
        async with async_session_factory() as session:
            await do_work(session)
            await session.commit()
    except asyncio.CancelledError:
        raise   # propagate cleanly to lifespan shutdown
    except Exception:
        log.exception(...)  # log and continue; never crash the loop
```

This means a transient DB error does not take down the API process.

---

## Redis Layout

| Key pattern | Type | TTL | Written by | Read by |
|---|---|---|---|---|
| `ros:plan:{name}:current` | String (JSON) | 30s | Query router (cache miss), Deployment service (on activate) | Query router |
| `ros:deployment:{name}:active` | String | 300s | Deployment service | Query router (Phase 4+) |
| `ros:qcache:{sha256}` | String (JSON) | plan's `cache_ttl_seconds` | Retrieval executor (cache set) | Retrieval executor (cache get) |

All Redis writes are fire-and-forget: failures are logged and swallowed, never surfaced to callers.

---

## Database Layout

```
retrieval_plans             — one row per named plan
plan_versions               — one row per config snapshot
deployments                 — one row per deploy attempt
usage_records               — one row per query (fire-and-forget insert)

[Phase 5]
lineage_artifacts           — dataset snapshots, embedding artifacts, index artifacts
lineage_edges               — DAG edges (parent → child)

[Phase 6]
eval_jobs                   — QUEUED/RUNNING/COMPLETED/FAILED
eval_runs                   — computed metric snapshots
regression_alerts           — fired alerts + acknowledgement state

[Phase 7]
cost_entries                — hourly aggregations of usage_records
model_pricing               — per-token pricing with effective date ranges
recommendations             — active cost/perf recommendations
```

### Primary key strategy: UUIDv7

All tables use UUIDv7 as primary keys. UUIDv7 is time-ordered (48-bit millisecond timestamp in the high bits), which:

- Keeps new rows physically adjacent on disk (reduces page splits)
- Embeds creation timestamp (no separate `created_at` needed for PK-based range scans)
- Is unguessable (unlike autoincrement)
- Is globally unique without a coordination step

```python
def uuid7() -> uuid.UUID:
    timestamp_ms = int(time.time() * 1000)     # 48 bits
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 62) & 0x0FFF              # 12 bits (version field)
    rand_b = rand & 0x3FFFFFFFFFFFFFFF          # 62 bits
    high = (timestamp_ms << 16) | 0x7000 | rand_a
    low = 0x8000000000000000 | rand_b
    return uuid.UUID(int=(high << 64) | low)
```

---

## Config Hash

`compute_config_hash()` produces a SHA-256 of the canonical JSON of the fields that define *retrieval behaviour*. Fields that only affect cost, caching, or governance are excluded.

**Included in hash:**
`embedding_provider`, `embedding_model`, `embedding_dimensions`, `modalities`, `index_backend`, `index_collection`, `distance_metric`, `quantization`, `top_k`, `rerank_top_k`, `reranker`, `hybrid_alpha`, `metadata_filters`

**Excluded from hash:**
`cache_ttl_seconds`, `cache_enabled`, `max_tokens_per_query`, `change_comment`, `created_by`

Lists are sorted before hashing so `["text", "image"]` and `["image", "text"]` produce the same hash.

The hash serves two purposes:
1. **Deduplication** — creating a version with identical retrieval behaviour to an existing version returns HTTP 409 (`DUPLICATE_CONFIG_HASH`) without a DB write.
2. **Cross-plan eval sharing** — two plans with the same `config_hash` necessarily produce identical retrieval results. Eval runs can be shared (Phase 6).

---

## Deployment State Machine

```
        ┌─────────────────────────────┐
        ▼                             │  manual rollback
    PENDING ──► ROLLING_OUT ──► ACTIVE
                                  │
                                  └──► ROLLING_BACK ──► ROLLED_BACK
                                  │
                                  └──[unrecoverable error]──► FAILED
```

State transitions:

| From | To | Trigger | Guard |
|---|---|---|---|
| `PENDING` | `ACTIVE` | `create_deployment` (no rollout params) | No other live deployment for this plan |
| `PENDING` | `ROLLING_OUT` | `create_deployment` (with rollout params) | No other live deployment for this plan |
| `ROLLING_OUT` | `ACTIVE` | rollout stepper (traffic_weight reaches 1.0) | Automated |
| `ACTIVE` | `ROLLING_BACK` | `POST /rollback` or rollback watchdog | Deployment must be live |
| `ROLLING_BACK` | `ROLLED_BACK` | Immediate (same request) | — |

**Live** = `ACTIVE` or `ROLLING_OUT`. At most one live deployment per plan is enforced at the service layer.

### Gradual rollout

When `rollout_step_percent` and `rollout_step_interval_seconds` are provided, the rollout stepper background loop advances `traffic_weight` by `rollout_step_percent / 100` on each run interval until it reaches `1.0`:

```
Step 0:  traffic_weight = 0.00   (ROLLING_OUT)
Step 1:  traffic_weight = 0.10
Step 2:  traffic_weight = 0.20
...
Step 10: traffic_weight = 1.00   (→ ACTIVE, activated_at set)
```

---

## Semantic Query Cache

Cache key = `SHA-256(plan_name | version_num | query_text | top_k)`

Cache miss flow:
```
1. Redis GET  →  miss
2. Embed query
3. Qdrant ANN search
4. (optional rerank)
5. Redis SET  with ex=cache_ttl_seconds
6. Return results
```

Cache hit flow:
```
1. Redis GET  →  hit  →  return immediately
```

Cache is bypassed when `cache_enabled = False` or `cache_ttl_seconds = 0`.

Cache entries are keyed per **plan version**. Deploying a new plan version automatically invalidates because the version number is part of the key — no explicit flush needed.

---

## Observability Stack

```
FastAPI ──► OTel SDK ──► OTLP/gRPC ──► Jaeger  (traces)
        └─► prometheus_client ──────────► Prometheus ──► Grafana  (metrics)
        └─► structlog ──────────────────► stdout (JSON logs)
```

Every query generates:
- A root OTel span (`retrieval_os.query`) with child spans for cache, embed, and index
- A histogram observation on `retrieval_os_retrieval_latency_seconds`
- A counter increment on `retrieval_os_retrieval_requests_total`
- A fire-and-forget `usage_records` insert

See [observability.md](./observability.md) for the complete metric catalogue and trace structure.

---

## Error Hierarchy

All errors inherit from `RetrievalOSError` which carries `status_code`, `error_code` (machine-readable), `message` (human-readable), and `detail` (structured context). The global exception handler returns:

```json
{
  "error": "PLAN_NOT_FOUND",
  "message": "Plan 'my-docs' not found",
  "detail": {}
}
```

| Class | HTTP | error_code |
|---|---|---|
| `PlanNotFoundError` | 404 | `PLAN_NOT_FOUND` |
| `PlanVersionNotFoundError` | 404 | `PLAN_VERSION_NOT_FOUND` |
| `DeploymentNotFoundError` | 404 | `DEPLOYMENT_NOT_FOUND` |
| `ConflictError` | 409 | `CONFLICT` |
| `DuplicateConfigError` | 409 | `DUPLICATE_CONFIG_HASH` |
| `DeploymentStateError` | 409 | `DEPLOYMENT_STATE_ERROR` |
| `AppValidationError` | 422 | `VALIDATION_ERROR` |
| `IndexBackendError` | 503 | `INDEX_BACKEND_ERROR` |
| `EmbeddingProviderError` | 503 | `EMBEDDING_PROVIDER_ERROR` |

Validation errors include a `detail.errors` list with one message per failed field, so callers see all problems at once rather than iterating through fix-and-retry cycles.
