# Data Models

All tables use UUIDv7 primary keys unless noted. UUIDv7 is time-ordered (48-bit millisecond timestamp in the most-significant bits), which keeps rows clustered on disk and embeds creation time without a separate column.

---

## `projects`

A named container for a retrieval use case (e.g. "docs-search", "product-catalog"). Holds metadata and acts as the namespace for index configs, deployments, ingestion jobs, and eval runs.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `name` | VARCHAR(255) | No | — | Globally unique slug. Lowercase letters, digits, hyphens. Cannot start/end with hyphen. |
| `description` | TEXT | No | `''` | Free-text description. |
| `is_archived` | BOOLEAN | No | `false` | Soft-delete flag. Archived projects cannot receive new index configs or deployments. |
| `created_at` | TIMESTAMPTZ | No | — | Creation timestamp. |
| `updated_at` | TIMESTAMPTZ | No | — | Last update timestamp (updated on archive). |
| `created_by` | VARCHAR(255) | No | — | Identity string of creator. |

**Indexes:**
- UNIQUE on `name`
- on `created_at` (for list ordering)

---

## `index_configs`

An immutable snapshot of how to *build* the index: embedding model, vector dimensions, collection, and distance metric. Build-time settings only — changing any of these requires re-ingesting documents. The current config is identified by `is_current = true`.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `project_id` | UUID | No | — | FK → `projects.id` ON DELETE CASCADE. |
| `version` | INTEGER | No | — | Monotonically increasing per project. Assigned via `SELECT FOR UPDATE` on the parent project row. |
| `is_current` | BOOLEAN | No | `false` | True for the latest version only. Previous version's `is_current` is set to false on new version creation (within the same transaction). |
| **Embedding config** | | | | |
| `embedding_provider` | VARCHAR(100) | No | — | One of: `sentence_transformers`, `openai`, `clip`, `whisper`, `video_frame`. |
| `embedding_model` | VARCHAR(255) | No | — | Provider-specific model identifier. |
| `embedding_dimensions` | INTEGER | No | — | Output vector dimensionality. |
| `modalities` | TEXT[] | No | — | PostgreSQL array. At least one element. |
| `embedding_batch_size` | INTEGER | No | 32 | Number of texts per embed call. |
| `embedding_normalize` | BOOLEAN | No | `true` | L2-normalise output vectors. |
| **Index config** | | | | |
| `index_backend` | VARCHAR(100) | No | — | One of: `qdrant`, `pgvector`. |
| `index_collection` | VARCHAR(255) | No | — | Collection/table name in the index. |
| `distance_metric` | VARCHAR(50) | No | — | One of: `cosine`, `dot`, `euclidean`. |
| `quantization` | VARCHAR(50) | Yes | `NULL` | One of: `scalar`, `product`, null. |
| **Governance** | | | | |
| `change_comment` | TEXT | No | `''` | Human-readable note about this version. |
| `created_at` | TIMESTAMPTZ | No | — | Version creation timestamp. |
| `created_by` | VARCHAR(255) | No | — | Identity of version creator. |
| `config_hash` | CHAR(64) | No | — | SHA-256 hex digest of canonical index config. See below. |

**Constraints:**
- `uq_index_config_version` — UNIQUE (`project_id`, `version`)
- `uq_index_config_hash` — UNIQUE (`project_id`, `config_hash`) — prevents duplicate index configs within a project

**Indexes:**
- on `project_id`
- on (`project_id`, `is_current`)

### `config_hash` computation

SHA-256 of a canonical JSON string built from index-relevant fields only (search config fields live on `deployments` and are excluded):

```
embedding_provider, embedding_model, embedding_dimensions,
modalities (sorted), index_backend, index_collection,
distance_metric, quantization
```

The JSON is serialised with `sort_keys=True` and no whitespace. Lists are sorted before serialisation so field order does not affect the hash.

---

## `deployments`

A deployment binds an IndexConfig to live traffic, carries the runtime search config, and tracks the rollout lifecycle.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | UUIDv7 | Primary key (stored as string for portability). |
| `plan_name` | VARCHAR(255) | No | — | Denormalised project name (avoid join on hot queries). |
| `index_config_id` | UUID | No | — | FK → `index_configs.id`. Which index this deployment serves. |
| `index_config_version` | INTEGER | No | — | Denormalised version number for display. |
| `status` | VARCHAR(50) | No | `'PENDING'` | State machine value. See status table below. |
| `traffic_weight` | FLOAT | No | `0.0` | Fraction of live traffic (0.0–1.0). |
| `rollout_step_percent` | FLOAT | Yes | `NULL` | Traffic increment per rollout step (1–100). NULL = instant deploy. |
| `rollout_step_interval_seconds` | INTEGER | Yes | `NULL` | Seconds between rollout steps. |
| `rollback_recall_threshold` | FLOAT | Yes | `NULL` | Auto-rollback if Recall@5 drops below this. |
| `rollback_error_rate_threshold` | FLOAT | Yes | `NULL` | Auto-rollback if error rate exceeds this. |
| `eval_dataset_uri` | TEXT | Yes | `NULL` | S3 URI of JSONL eval dataset. If set, an EvalJob is automatically queued when this deployment is activated. |
| **Search config** | | | | |
| `top_k` | INTEGER | No | 10 | Number of ANN candidates returned from the index. |
| `rerank_top_k` | INTEGER | Yes | `NULL` | Post-rerank result count. Must be ≤ `top_k` if set. |
| `reranker` | VARCHAR(255) | Yes | `NULL` | Reranker identifier (`"provider:model"` format). |
| `hybrid_alpha` | FLOAT | Yes | `NULL` | Dense/sparse blend weight (0.0–1.0). |
| `metadata_filters` | JSONB | Yes | `NULL` | Default filter applied to all queries for this deployment. |
| `tenant_isolation_field` | VARCHAR(255) | Yes | `NULL` | Payload field used to scope queries by tenant. |
| `cache_enabled` | BOOLEAN | No | `true` | Enable semantic query cache. |
| `cache_ttl_seconds` | INTEGER | No | 3600 | Cache TTL in seconds. 0 = disabled. |
| `max_tokens_per_query` | INTEGER | Yes | `NULL` | Token cap for context packing. |
| **Governance** | | | | |
| `change_note` | TEXT | No | `''` | Human-readable deployment note. |
| `created_at` | TIMESTAMPTZ | No | — | Deployment creation time. |
| `updated_at` | TIMESTAMPTZ | No | — | Last status update time. |
| `created_by` | VARCHAR(255) | No | — | Identity of deployer. |
| `activated_at` | TIMESTAMPTZ | Yes | `NULL` | When status first became ACTIVE. |
| `rolled_back_at` | TIMESTAMPTZ | Yes | `NULL` | When rollback completed. |
| `rollback_reason` | TEXT | Yes | `NULL` | Reason string from rollback request or watchdog. |

**Status values:**

| Value | Meaning |
|---|---|
| `PENDING` | Created, not yet live. |
| `ROLLING_OUT` | Live with traffic_weight < 1.0. Rollout stepper advances it. |
| `ACTIVE` | Fully live at 100% traffic. |
| `ROLLING_BACK` | Rollback in progress (transitions to ROLLED_BACK immediately). |
| `ROLLED_BACK` | No longer live. traffic_weight = 0. |
| `FAILED` | Unrecoverable error. |

**Indexes:**
- `idx_deployments_plan_name` — on `plan_name`
- `idx_deployments_status` — on (`plan_name`, `status`) — for "get active deployment" query
- `idx_deployments_created_at` — on `created_at`

---

## `usage_records`

One row per query. Written asynchronously via `asyncio.create_task()` — never on the critical path.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | UUIDv7 | Primary key. |
| `plan_name` | VARCHAR(255) | No | — | Plan name queried. |
| `plan_version` | INTEGER | No | — | Plan version that served the query. |
| `query_chars` | INTEGER | No | — | Number of characters in the query string. |
| `result_count` | INTEGER | No | — | Number of chunks returned. |
| `cache_hit` | BOOLEAN | No | — | Whether the response was served from cache. |
| `latency_ms` | FLOAT | No | — | End-to-end query latency in milliseconds (from request receipt to response ready). |
| `created_at` | TIMESTAMPTZ | No | — | Query timestamp. |

**Indexes:**
- `idx_usage_records_plan_name` — on (`plan_name`, `created_at`) — for per-plan time-range queries
- `idx_usage_records_created_at` — on `created_at` — for global time-range queries

Phase 7 aggregates `usage_records` into `cost_entries` hourly.

---

## `lineage_artifacts`

Tracks dataset snapshots, embedding artifacts, and index artifacts as nodes in the lineage DAG.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `artifact_type` | VARCHAR(50) | No | — | One of: `DATASET_SNAPSHOT`, `EMBEDDING_ARTIFACT`, `INDEX_ARTIFACT`. |
| `name` | VARCHAR(255) | No | — | Human-readable name. |
| `version` | VARCHAR(100) | No | — | Semantic version or git SHA. |
| `storage_uri` | TEXT | No | — | `s3://bucket/path` or `qdrant://collection`. Unique per artifact. |
| `content_hash` | CHAR(64) | Yes | `NULL` | SHA-256 of artifact bytes (populated when available). |
| `metadata` | JSONB | Yes | `NULL` | Type-specific metadata (e.g. chunk count, vector dimensions). |
| `created_at` | TIMESTAMPTZ | No | — | Registration timestamp. |
| `created_by` | VARCHAR(255) | No | — | Identity of registrar. |

**Indexes:**
- `idx_lineage_artifacts_uri` — UNIQUE on `storage_uri`
- `idx_lineage_artifacts_type` — on `artifact_type`

---

## `lineage_edges`

DAG edges between artifacts.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `parent_artifact_id` | UUID | No | — | FK → `lineage_artifacts.id`. |
| `child_artifact_id` | UUID | No | — | FK → `lineage_artifacts.id`. |
| `relationship` | VARCHAR(50) | No | — | One of: `produced_from`, `derived_from`, `deployed_as`. |
| `created_at` | TIMESTAMPTZ | No | — | Edge creation timestamp. |

**Constraints:**
- `uq_lineage_edge` — UNIQUE (`parent_artifact_id`, `child_artifact_id`, `relationship`) — prevents duplicate edges.

Cycles are prevented at the service layer: an ancestor query is run before any edge insert and raises `ConflictError` if a cycle would be created.

---

## `eval_jobs`

Tracks evaluation runs against a plan version. Each job scores retrieval quality using a JSONL ground-truth file and writes metrics to `eval_runs`.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `plan_name` | VARCHAR(255) | No | — | Plan being evaluated. |
| `plan_version` | INTEGER | No | — | Plan version being evaluated. |
| `status` | VARCHAR(50) | No | `'QUEUED'` | `QUEUED`, `RUNNING`, `COMPLETED`, or `FAILED`. |
| `ground_truth_uri` | TEXT | No | — | S3 URI of JSONL ground-truth file. Each line: `{"query": "...", "relevant_ids": [...]}`. |
| `sample_count` | INTEGER | No | — | Number of queries to evaluate (sampled from ground truth). |
| `recall_at_5` | FLOAT | Yes | `NULL` | Recall@5 metric. Populated on COMPLETED. |
| `mrr` | FLOAT | Yes | `NULL` | Mean Reciprocal Rank. |
| `ndcg_at_10` | FLOAT | Yes | `NULL` | NDCG@10. |
| `total_queries` | INTEGER | Yes | `NULL` | Number of queries executed. |
| `failed_queries` | INTEGER | Yes | `NULL` | Queries that raised exceptions during eval. |
| `regression_detected` | BOOLEAN | No | `false` | True if recall degraded vs. the previous completed eval for this plan. |
| `created_at` | TIMESTAMPTZ | No | — | Job submission timestamp. |
| `started_at` | TIMESTAMPTZ | Yes | `NULL` | When the background runner claimed the job. |
| `completed_at` | TIMESTAMPTZ | Yes | `NULL` | When the job reached COMPLETED or FAILED. |
| `error_message` | TEXT | Yes | `NULL` | Failure reason if FAILED. |
| `created_by` | VARCHAR(255) | Yes | `NULL` | Identity of submitter. |

**Indexes:**
- `idx_eval_jobs_plan` — on (`plan_name`, `plan_version`, `created_at`)
- `idx_eval_jobs_status` — on (`status`, `created_at`) — for the background job-runner poll query

---

## `cost_entries`

Hourly aggregation of `usage_records`. Written by the `cost_aggregator` background loop.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `plan_name` | VARCHAR(255) | No | — | Plan name. |
| `plan_version` | INTEGER | No | — | Plan version. |
| `window_start` | TIMESTAMPTZ | No | — | Hour boundary (`date_trunc('hour', ...)`). |
| `query_count` | INTEGER | No | — | Total queries in the window. |
| `cache_hit_count` | INTEGER | No | — | Cache hits in the window. |
| `total_chars` | BIGINT | No | — | Sum of `usage_records.query_chars`. |
| `estimated_cost_usd` | FLOAT | No | — | Estimated cost based on model pricing rules. |
| `created_at` | TIMESTAMPTZ | No | — | When this aggregate was computed. |

**Constraints:**
- `uq_cost_entry_window` — UNIQUE (`plan_name`, `plan_version`, `window_start`) — makes aggregation idempotent.

---

## `tenants`

Tenants are logical isolation units. Every API request authenticated with an API key is scoped to a tenant.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `name` | VARCHAR(255) | No | — | Globally unique tenant name. |
| `description` | TEXT | No | `''` | Free-text description. |
| `is_active` | BOOLEAN | No | `true` | Inactive tenants are rejected at auth. |
| `rate_limit_rpm` | INTEGER | No | `60` | Max requests per minute (Redis sliding-window). |
| `created_at` | TIMESTAMPTZ | No | — | Creation timestamp. |
| `updated_at` | TIMESTAMPTZ | No | — | Last update timestamp. |

**Indexes:**
- `idx_tenants_name` — UNIQUE on `name`

---

## `api_keys`

Hashed API keys for tenant authentication.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `tenant_id` | UUID | No | — | FK → `tenants.id` ON DELETE CASCADE. |
| `key_hash` | CHAR(64) | No | — | SHA-256 hex digest of the raw API key. The raw key is never stored. |
| `description` | VARCHAR(255) | Yes | `NULL` | Human-readable label (e.g. `"CI pipeline key"`). |
| `is_active` | BOOLEAN | No | `true` | Inactive keys are rejected at auth. |
| `last_used_at` | TIMESTAMPTZ | Yes | `NULL` | Updated on each authenticated request. |
| `created_at` | TIMESTAMPTZ | No | — | Creation timestamp. |

**Indexes:**
- `idx_api_keys_key_hash` — UNIQUE on `key_hash` — supports O(1) lookup at auth time
- `idx_api_keys_tenant_id` — on `tenant_id`

---

## `webhook_subscriptions`

Outbound HTTP event subscriptions. Each row registers a URL to receive signed event payloads.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | UUIDv7 | Primary key. |
| `url` | VARCHAR(2048) | No | — | Target HTTP(S) URL. |
| `events` | JSON | No | `[]` | List of event type strings to match. Empty = all events. |
| `secret` | TEXT | Yes | `NULL` | Shared secret for HMAC-SHA256 signing. Never returned by the API. |
| `description` | TEXT | Yes | `NULL` | Human-readable label. |
| `is_active` | BOOLEAN | No | `true` | Inactive subscriptions receive no deliveries. |
| `created_at` | TIMESTAMPTZ | No | — | Registration timestamp. |
| `updated_at` | TIMESTAMPTZ | No | — | Last update timestamp. |

**Indexes:**
- `idx_webhook_subscriptions_is_active` — on `is_active` — for the active-subscription query on each event dispatch

---

## `ingestion_jobs`

Tracks document ingestion jobs. Each job chunks, embeds, and upserts documents into the IndexConfig's vector collection.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | UUIDv7 | Primary key. |
| `plan_name` | VARCHAR(255) | No | — | Target project name. |
| `index_config_id` | UUID | No | — | FK → `index_configs.id`. Which index is populated. |
| `index_config_version` | INTEGER | No | — | Denormalised version number for display. |
| `source_uri` | TEXT | Yes | `NULL` | S3 URI of a JSONL source file. Null if documents were provided inline. |
| `document_payload` | JSON | Yes | `NULL` | Inline document list serialised as JSON. Null if `source_uri` is set. |
| `chunk_size` | INTEGER | No | 512 | Max words per chunk. |
| `overlap` | INTEGER | No | 64 | Word overlap between consecutive chunks. |
| `status` | VARCHAR(50) | No | `'QUEUED'` | `QUEUED`, `RUNNING`, `COMPLETED`, or `FAILED`. |
| `total_docs` | INTEGER | Yes | `NULL` | Total documents processed. |
| `total_chunks` | INTEGER | Yes | `NULL` | Total chunks generated. |
| `indexed_chunks` | INTEGER | Yes | `NULL` | Chunks successfully upserted to Qdrant. |
| `failed_chunks` | INTEGER | Yes | `NULL` | Chunks that failed to embed or upsert. |
| `duplicate_of` | VARCHAR(36) | Yes | `NULL` | ID of a previously COMPLETED job with the same `(project_name, index_config_version)`. Set when re-embedding is skipped; stats are copied from the original job rather than recomputed. |
| `error_message` | TEXT | Yes | `NULL` | Set if status is FAILED. |
| `created_at` | TIMESTAMPTZ | No | — | Job submission timestamp. |
| `started_at` | TIMESTAMPTZ | Yes | `NULL` | When the background runner claimed the job. |
| `completed_at` | TIMESTAMPTZ | Yes | `NULL` | When the job finished. |
| `created_by` | VARCHAR(255) | Yes | `NULL` | Identity of submitter. |

**Indexes:**
- `idx_ingestion_jobs_status_created` — on (`status`, `created_at`) — for the background job-runner poll query (SELECT FOR UPDATE SKIP LOCKED)
- `idx_ingestion_jobs_plan` — on (`plan_name`, `created_at`)

---

## ORM Patterns

All ORM models use SQLAlchemy 2.0 `Mapped[]` annotations:

```python
from sqlalchemy.orm import Mapped, mapped_column

class MyModel(Base):
    __tablename__ = "my_table"

    id: Mapped[str] = mapped_column(
        PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid7())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
```

Relationships use `selectin` loading to avoid N+1 on list queries:

```python
index_configs: Mapped[list[IndexConfig]] = relationship(
    "IndexConfig",
    back_populates="project",
    lazy="selectin",
    order_by="IndexConfig.version",
)
```

Session management via FastAPI dependency injection:

```python
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

The `expire_on_commit=False` option on `async_session_factory` means ORM objects remain accessible after `commit()` without triggering a lazy reload — important for response serialisation.
