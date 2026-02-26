# Architecture

Retrieval-OS is a production control plane for RAG and semantic search systems. It sits between your application and your vector database, and provides versioning, deployment control, quality measurement, and automatic rollback for retrieval pipelines — the operational controls that vector databases and embedding APIs do not provide on their own.

---

## Core Concepts

Before reading the technical design, it helps to understand what each concept maps to from a user's perspective.

### Project

A **Project** is a named container for a retrieval use case — think "docs-search", "product-catalog", or "support-tickets". It holds metadata and provides the namespace under which all index configs, deployments, ingestion jobs, and eval runs are scoped.

### IndexConfig

An **IndexConfig** is an immutable, versioned snapshot of how to *build* the index: embedding model, vector dimensions, which collection to use, distance metric, quantization. These are the build-time settings. Changing any of them requires re-ingesting documents into the new index. Versions are numbered sequentially and serve as the shared reference between ingestion and the serving path — the embedding model that encodes documents at ingest time must match the model that encodes queries at retrieval time.

### Deployment

A **Deployment** activates an index config for live traffic and carries the *search config* — the runtime tuning that does not touch the index: `top_k`, `reranker`, `hybrid_alpha`, cache settings, metadata filters. You can deploy instantly (100% traffic switch) or gradually (start at a low traffic weight, advance automatically on a schedule, promote to full traffic when stable). At most one deployment is live per project at any time.

Deployments separate "what index" (IndexConfig FK) from "how to search it" (search config on the Deployment itself). Two deployments can reference the same IndexConfig with different top-k or reranker configs without re-indexing.

### Ingestion Job

An **Ingestion Job** loads documents into the vector index using a specific IndexConfig's embedding settings. You push documents via API; the system chunks them, embeds them using the IndexConfig's model, and upserts the vectors into the correct collection. The ingestion job records which IndexConfig version processed each document, so the index always has a clear provenance chain.

### Evaluation Job

An **Evaluation Job** measures retrieval quality for a project. You provide labelled query–document pairs (ground truth); the system runs each query through the live retrieval pipeline and computes Recall@k, MRR, and NDCG. Results are compared to the previous evaluation's baseline. A drop beyond a configurable threshold fires a regression alert.

### Deployment Guard-Rails

When creating a deployment, you can attach quality thresholds: a minimum Recall@5, a maximum error rate, or both. The rollback watchdog checks active deployments against the latest completed evaluation. If a threshold is breached, the deployment is rolled back automatically — no human required.

#### Automatic Quality Gating

Setting `eval_dataset_uri` on a deployment closes the quality loop automatically:

1. **Activate** — `POST /v1/projects/{name}/deployments` with `eval_dataset_uri` set.
2. **Auto-eval** — On activation the service calls `auto_queue_eval()`, which creates a QUEUED `EvalJob` for the same `index_config_version`. This is **best-effort**: if the queue call fails (e.g., DB error) the deployment still becomes ACTIVE.
3. **Eval runner** — The background `eval_job_runner` loop picks up the job, runs retrieval against the eval dataset, and writes back Recall@k / MRR / NDCG metrics.
4. **Watchdog** — The `rollback_watchdog` loop reads the completed eval result and compares it against the deployment's `rollback_recall_threshold` / `rollback_error_rate_threshold`. If a threshold is breached, the deployment is automatically rolled back.

Without `eval_dataset_uri`, eval jobs must be queued manually via `POST /v1/eval/jobs`.

#### Query Timeout

Every query through `route_query()` is wrapped with `asyncio.wait_for(timeout=settings.query_timeout_seconds)` (default: 30 s). If Qdrant or the embedding backend hangs past this deadline, the call is cancelled and the API returns HTTP 504 with `{"error": "QUERY_TIMEOUT"}`. Configure via the `QUERY_TIMEOUT_SECONDS` environment variable.

The timeout applies to the full retrieval pipeline (embed + ANN + cache write). The circuit breaker (`CircuitOpenError`) fires independently on repeated upstream failures and returns HTTP 503.

#### Ingestion Deduplication

Before embedding a new ingestion job, the runner checks for an existing COMPLETED job for the same `(project_name, index_config_version)`. If one exists the new job is marked COMPLETED immediately, with `duplicate_of` set to the prior job's id and counters copied from it. No embedding or Qdrant upsert is performed.

**"Same config version"** means the same `index_config_version` integer for the same project — i.e., re-POSTing `/ingest` with the same documents or a different document set against the same IndexConfig version. To force a re-index, create a new IndexConfig version (which increments the version number) and submit a new ingestion job for that version.

### Lineage

Every artifact produced by the system — a chunked dataset, an embedding run, an index snapshot — is recorded as a node in a lineage graph, with directed edges linking parents to children. You can trace any document chunk in the index back to its source file, the embedding model that encoded it, and the IndexConfig version that owns it.

---

## The Core Problem

Retrieval-augmented systems break in silent ways. You swap an embedding model, rebuild your index with slightly different settings, or adjust top-k — and recall drops 12% a week later with no clear cause. There is no equivalent of "git diff" for retrieval config. No controlled rollout. No automatic quality regression detection.

Retrieval-OS solves this by:

1. Separating *index config* (build-time) from *search config* (runtime) into distinct versioned objects
2. Treating index changes as **Deployments** with traffic weights and rollback capability
3. Measuring **Recall@k, MRR, NDCG** continuously against a held-out ground truth
4. Tracking the full **lineage** from training dataset to deployed index
5. Aggregating **cost and latency** so infrastructure decisions are evidence-based

---

## Three Paths Through the System

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   API Gateway (FastAPI)                                       │
│  /v1/query/{name}  /v1/projects/{name}/ingest  /v1/projects  /v1/projects/{n}/deployments    │
│  /health           /ready                      /metrics      /v1/info                        │
└────────┬─────────────────────┬──────────────────────────┬────────────────────────────────────┘
         │                     │                          │
  ╔══════▼══════╗       ╔══════▼══════╗           ╔══════▼══════╗
  ║ SERVING PATH ║       ║  INGESTION  ║           ║ MANAGEMENT  ║
  ║  P99 < 200ms ║       ║    PATH     ║           ║    PATH     ║
  ╚══════╤══════╝       ╚══════╤══════╝           ╚══════╤══════╝
         │                     │                          │
  Query Router          Creates QUEUED            Project Manager
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
POST /v1/query/{project_name}
        │
        ▼
 Redis GET ros:project:{name}:active   ← cache miss → Postgres read (active deployment + IndexConfig), then cache
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

- **No Postgres reads on the hot path.** Serving config lives in Redis (`ros:project:{name}:active`, 30s TTL). A Postgres failure cannot cause query failures. On Redis miss, one Postgres read (active deployment + its IndexConfig) warms the cache.
- **Config validation at write time.** Every `IndexConfig` and `Deployment` row in Postgres is guaranteed valid by the service layer. The serving path performs zero defensive checks.
- **Usage records are non-blocking.** After every query, a fire-and-forget `asyncio.create_task()` writes a `usage_record` row. The response is returned before this completes.

---

## Ingestion Path

Documents are loaded into the vector index asynchronously via a background job runner.

```
POST /v1/projects/{name}/ingest
        │
        ▼
 Validate IndexConfig version exists (Postgres)
        │
        ▼
 Create IngestionJob (status: QUEUED)
        │
        │  ← HTTP 202 returned here — job runs in background
        │
        │  ingestion_job_runner polls every 5s
        │  SELECT FOR UPDATE SKIP LOCKED on ingestion_jobs WHERE status='QUEUED'
        ▼
 1. Load IndexConfig from Postgres
        │
        ▼
 2. Load documents
    ├── inline: JSON array in request body
    └── S3:     download JSONL from s3://bucket/key
        │
        ▼
 3. Chunk each document (word-boundary)
    chunk_size and overlap come from the request (not the IndexConfig)
    each chunk carries: doc_id, chunk_idx, plan_name, index_config_version, doc metadata
        │
        ▼
 4. For each batch of 50 chunks:
    ├── embed_text → vectors  (uses IndexConfig's provider + model)
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
 6. Fire webhook (ingestion.completed) if indexed_chunks > 0
        │
        ▼
 7. Mark job COMPLETED (indexed_chunks, failed_chunks, total_chunks recorded)
    [unrecoverable error at any step → FAILED with error_message]
```

### Ingestion path properties

- **Per-batch error isolation.** An embed or upsert failure on one batch does not abort the job. That batch's chunks are counted as `failed_chunks`; the remaining batches continue. The job reaches `COMPLETED` even if some chunks failed — allowing partial indexing.
- **Collection auto-creation.** `ensure_collection` is called exactly once per job (before the first upsert), using the actual vector dimension inferred from the first embed result. If the collection already exists it is a no-op.
- **Chunk payloads are queryable.** Every vector in Qdrant carries a payload with `doc_id`, `chunk_idx`, `plan_name`, `index_config_version`, and all metadata from the source document. These are filterable at query time.
- **Lineage is idempotent.** Re-running a job for the same IndexConfig version does not create duplicate lineage artifacts. Each artifact is keyed by its storage URI and skipped if it already exists.
- **Horizontal scaling.** `SELECT FOR UPDATE SKIP LOCKED` means multiple API replicas can each run an ingestion runner without processing the same job twice.

---

## Management Path

The management path handles all writes to plans, deployments, and configuration. These operations are infrequent relative to queries and have no latency SLO.

### Management path invariants

- **IndexConfig version numbers are monotonic and gapless.** `SELECT FOR UPDATE` on the parent `Project` row serialises concurrent index config creates.
- **Identical index configs are rejected.** `config_hash` = SHA-256 of index-relevant fields only. Duplicate hash → HTTP 409 before DB write.
- **At most one live deployment per project.** The service layer enforces this; there is no DB unique constraint to race against because the check + insert happen within a transaction with the parent project row locked.

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
| `ros:project:{name}:active` | String (JSON) | 30s | Query router (cache miss), Deployment service (on activate) | Query router |
| `ros:deployment:{name}:active` | String | 300s | Deployment service | Query router (cache miss fallback) |
| `ros:qcache:{sha256}` | String (JSON) | deployment's `cache_ttl_seconds` | Retrieval executor (cache set) | Retrieval executor (cache get) |

All Redis writes are fire-and-forget: failures are logged and swallowed, never surfaced to callers.

---

## Database Layout

```
projects                    — one row per named retrieval project
index_configs               — one row per immutable index config snapshot (build-time)
deployments                 — one row per deploy attempt (search config + index FK + lifecycle)
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

**Included in hash (index-only fields):**
`embedding_provider`, `embedding_model`, `embedding_dimensions`, `modalities`, `index_backend`, `index_collection`, `distance_metric`, `quantization`

**Excluded from hash (now on Deployment or governance-only):**
`top_k`, `rerank_top_k`, `reranker`, `hybrid_alpha`, `metadata_filters`, `cache_ttl_seconds`, `cache_enabled`, `max_tokens_per_query`, `change_comment`, `created_by`

Lists are sorted before hashing so `["text", "image"]` and `["image", "text"]` produce the same hash.

The hash serves two purposes:
1. **Deduplication** — creating an IndexConfig with identical index settings to an existing one returns HTTP 409 (`DUPLICATE_CONFIG_HASH`) without a DB write.
2. **Cross-project eval sharing** — two projects with the same `config_hash` necessarily use identical indexes. Eval runs can be shared.

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

Cache entries are keyed per **index config version**. Deploying a new IndexConfig version automatically invalidates because the version number is part of the key — no explicit flush needed.

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
  "error": "PROJECT_NOT_FOUND",
  "message": "Project 'my-docs' not found",
  "detail": {}
}
```

| Class | HTTP | error_code |
|---|---|---|
| `ProjectNotFoundError` | 404 | `PROJECT_NOT_FOUND` |
| `IndexConfigNotFoundError` | 404 | `INDEX_CONFIG_NOT_FOUND` |
| `DeploymentNotFoundError` | 404 | `DEPLOYMENT_NOT_FOUND` |
| `ConflictError` | 409 | `CONFLICT` |
| `DuplicateConfigError` | 409 | `DUPLICATE_CONFIG_HASH` |
| `DeploymentStateError` | 409 | `DEPLOYMENT_STATE_ERROR` |
| `AppValidationError` | 422 | `VALIDATION_ERROR` |
| `IndexBackendError` | 503 | `INDEX_BACKEND_ERROR` |
| `EmbeddingProviderError` | 503 | `EMBEDDING_PROVIDER_ERROR` |

Validation errors include a `detail.errors` list with one message per failed field, so callers see all problems at once rather than iterating through fix-and-retry cycles.


---

## Test Architecture

Retrieval-OS has four test layers, each proving a distinct class of correctness.

| Layer | Location | What it proves |
|---|---|---|
| Unit | `tests/unit/` | Correctness of individual functions: service logic, validators, state machines, schema parsing |
| Microbenchmarks | `tests/benchmarks/` | In-process CPU overhead of the serving hot-path (JSON decode, key derivation, config merge, metric computation) — no I/O, no infrastructure |
| E2E failure modes | `tests/e2e/` | Correct behaviour under adverse conditions with real Postgres/Redis: unknown project → 404, no active deployment → fast fail, watchdog skips healthy deployments, watchdog rolls back on recall breach |
| Load | `tests/load/` | Real system throughput and latency (real Qdrant + Redis + Postgres): QPS, p99, sustained stability, Zipf cache hit rate, zero-downtime upgrade, rollback speed, ingestion dedup, SLA timeout enforcement |

**Microbenchmarks** run purely in-process — no sockets, no threads, no filesystem. They establish a CPU budget: if `< 0.05 ms/query` overhead is demonstrated here, a 10 k QPS serving target adds less than 500 ms of pure Python overhead per CPU core.

**E2E failure mode tests** require `make infra && make migrate` (Postgres + Redis only; Qdrant not needed). They test the branches that are hard to exercise with mocks: SELECT FOR UPDATE SKIP LOCKED fairness, Redis cache invalidation on rollback, watchdog triggering rollback on a breached threshold.

**Load tests** require the full stack (`make infra` including Qdrant). Embedding is intentionally stubbed with random unit vectors throughout — this isolates the infrastructure latency that Retrieval-OS controls. Real embedding latency (2–150 ms depending on model and hardware) is additive and documented separately.
