# Architecture

Retrieval-OS is a serving layer that sits between your application and the underlying vector databases and embedding model APIs. It makes retrieval systems coherent, observable, upgradeable, and safe to change in production.

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

## Two Paths Through the System

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway (FastAPI)                        │
│       /v1/query    /v1/plans    /v1/plans/{n}/deployments        │
│       /health      /ready      /metrics      /v1/info            │
└──────────┬───────────────────────────┬───────────────────────────┘
           │                           │
    ╔══════▼══════╗             ╔══════▼══════╗
    ║ SERVING PATH ║             ║ MANAGEMENT  ║
    ║  P99 < 200ms ║             ║    PATH     ║
    ╚══════╤══════╝             ╚══════╤══════╝
           │                           │
    Query Router               Plan Manager
    (Redis config read)        (Postgres read/write)
           │
    Retrieval Executor         Deployment Controller
    (cache → embed → ANN)      (state machine + Redis weights)
           │
    Embed Router               Lineage Tracker       [Phase 5]
    (ST / OpenAI)              (DAG + orphan detect)
           │
    Index Proxy                Eval Engine           [Phase 6]
    (Qdrant gRPC)              (Recall@k, MRR, NDCG)
           │
    Redis Cache                Cost Intelligence     [Phase 7]
    (SHA-256 keyed)            (usage aggregation)
```

### Serving path invariants

These are not aspirational — they are enforced by construction:

- **No Postgres reads on the hot path.** The query router reads plan config from Redis (`ros:plan:{name}:current`, 30s TTL). A Postgres failure cannot cause query failures. On Redis miss, one Postgres read occurs and the result is immediately cached.
- **Plan validation at write time.** Every `PlanVersion` row in Postgres is guaranteed valid by the service layer. The serving path performs zero defensive checks on plan config.
- **Usage records are non-blocking.** After every query, a fire-and-forget `asyncio.create_task()` writes a `usage_record` row. The response is returned before this completes.

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
  ├── eval_job_runner()       every 5s       [activates Phase 6]
  │     SELECT FOR UPDATE SKIP LOCKED on eval_jobs WHERE status='QUEUED'.
  │     Processes one job per cycle. State survives restarts via DB status column.
  │
  └── cost_aggregator()       every 3600s    [activates Phase 7]
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
