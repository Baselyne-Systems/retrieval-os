# Data Models

All tables use UUIDv7 primary keys unless noted. UUIDv7 is time-ordered (48-bit millisecond timestamp in the most-significant bits), which keeps rows clustered on disk and embeds creation time without a separate column.

---

## `retrieval_plans`

A named container for a retrieval configuration. The plan itself holds metadata; all config lives in `plan_versions`.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `name` | VARCHAR(255) | No | — | Globally unique slug. Lowercase letters, digits, hyphens. Cannot start/end with hyphen. |
| `description` | TEXT | No | `''` | Free-text description. |
| `is_archived` | BOOLEAN | No | `false` | Soft-delete flag. Archived plans cannot receive new versions or deployments. |
| `created_at` | TIMESTAMPTZ | No | — | Creation timestamp. |
| `updated_at` | TIMESTAMPTZ | No | — | Last update timestamp (updated on archive). |
| `created_by` | VARCHAR(255) | No | — | Identity string of creator. |

**Indexes:**
- `idx_retrieval_plans_name` — UNIQUE on `name`
- `idx_retrieval_plans_created_at` — on `created_at` (for list ordering)

---

## `plan_versions`

An immutable retrieval config snapshot. Every change to embedding or index configuration creates a new row. The current version is identified by `is_current = true`.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | UUID | No | UUIDv7 | Primary key. |
| `plan_id` | UUID | No | — | FK → `retrieval_plans.id` ON DELETE CASCADE. |
| `version` | INTEGER | No | — | Monotonically increasing per plan. Assigned via `SELECT FOR UPDATE` on the parent plan row. |
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
| **Retrieval config** | | | | |
| `top_k` | INTEGER | No | — | Number of ANN candidates. |
| `rerank_top_k` | INTEGER | Yes | `NULL` | Post-rerank result count. Must be ≤ `top_k` if set. |
| `reranker` | VARCHAR(255) | Yes | `NULL` | Reranker model identifier. |
| `hybrid_alpha` | FLOAT | Yes | `NULL` | Dense/sparse blend (0.0–1.0). |
| **Filter config** | | | | |
| `metadata_filters` | JSONB | Yes | `NULL` | Default filter applied to all queries for this plan. |
| `tenant_isolation_field` | VARCHAR(255) | Yes | `NULL` | Payload field used to scope queries by tenant. |
| **Cost/cache config** | | | | |
| `cache_enabled` | BOOLEAN | No | `true` | Enable semantic query cache. |
| `cache_ttl_seconds` | INTEGER | No | 3600 | Cache TTL in seconds. 0 = disabled. |
| `max_tokens_per_query` | INTEGER | Yes | `NULL` | Token cap for context packing (Phase 8). |
| **Governance** | | | | |
| `change_comment` | TEXT | No | `''` | Human-readable note about this version. |
| `created_at` | TIMESTAMPTZ | No | — | Version creation timestamp. |
| `created_by` | VARCHAR(255) | No | — | Identity of version creator. |
| `config_hash` | CHAR(64) | No | — | SHA-256 hex digest of canonical config. See below. |

**Constraints:**
- `uq_plan_version` — UNIQUE (`plan_id`, `version`)
- `uq_plan_config_hash` — UNIQUE (`plan_id`, `config_hash`) — prevents duplicate retrieval behaviour within a plan

**Indexes:**
- `idx_plan_versions_plan_id` — on `plan_id`
- `idx_plan_versions_is_current` — on (`plan_id`, `is_current`)

### `config_hash` computation

SHA-256 of a canonical JSON string built from the following fields only (excludes cost/cache/governance fields):

```
embedding_provider, embedding_model, embedding_dimensions,
modalities (sorted), index_backend, index_collection,
distance_metric, quantization, top_k, rerank_top_k,
reranker, hybrid_alpha, metadata_filters
```

The JSON is serialised with `sort_keys=True` and no whitespace. Lists are sorted before serialisation so field order does not affect the hash.

---

## `deployments`

A deployment binds a plan version to live traffic and tracks its rollout lifecycle.

| Column | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `id` | VARCHAR(36) | No | UUIDv7 | Primary key (stored as string for portability). |
| `plan_name` | VARCHAR(255) | No | — | Denormalised plan name (avoid join on hot queries). |
| `plan_version` | INTEGER | No | — | Version number being deployed. |
| `status` | VARCHAR(50) | No | `'PENDING'` | State machine value. See status table below. |
| `traffic_weight` | FLOAT | No | `0.0` | Fraction of live traffic (0.0–1.0). |
| `rollout_step_percent` | FLOAT | Yes | `NULL` | Traffic increment per rollout step (1–100). NULL = instant deploy. |
| `rollout_step_interval_seconds` | INTEGER | Yes | `NULL` | Seconds between rollout steps. |
| `rollback_recall_threshold` | FLOAT | Yes | `NULL` | Auto-rollback if Recall@10 drops below this. |
| `rollback_error_rate_threshold` | FLOAT | Yes | `NULL` | Auto-rollback if error rate exceeds this. |
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

## Future Tables (Phase 5+)

### `lineage_artifacts` (Phase 5)

Tracks dataset snapshots, embedding artifacts, and index artifacts.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `artifact_type` | VARCHAR | `DATASET_SNAPSHOT`, `EMBEDDING_ARTIFACT`, `INDEX_ARTIFACT` |
| `name` | VARCHAR(255) | Human-readable name |
| `version` | VARCHAR(100) | Semantic version or git SHA |
| `storage_uri` | TEXT | `s3://bucket/path` or `qdrant://collection` |
| `content_hash` | CHAR(64) | SHA-256 of artifact bytes (populated on registration) |
| `metadata` | JSONB | Type-specific metadata (chunk count, dimension count, etc.) |
| `created_at` | TIMESTAMPTZ | — |
| `created_by` | VARCHAR(255) | — |

### `lineage_edges` (Phase 5)

DAG edges between artifacts.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `parent_artifact_id` | UUID | FK → `lineage_artifacts.id` |
| `child_artifact_id` | UUID | FK → `lineage_artifacts.id` |
| `relationship` | VARCHAR(50) | `produced_from`, `derived_from`, `deployed_as` |
| `created_at` | TIMESTAMPTZ | — |

Cycles are prevented at the service layer by running an ancestor query before any edge insert.

### `eval_jobs` (Phase 6)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `plan_name` | VARCHAR(255) | — |
| `plan_version` | INTEGER | — |
| `status` | VARCHAR(50) | `QUEUED`, `RUNNING`, `COMPLETED`, `FAILED` |
| `ground_truth_uri` | TEXT | S3 URI to JSONL ground truth file |
| `sample_count` | INTEGER | Number of queries to evaluate |
| `created_at` | TIMESTAMPTZ | — |
| `started_at` | TIMESTAMPTZ | Set when RUNNING |
| `completed_at` | TIMESTAMPTZ | Set when COMPLETED/FAILED |
| `error` | TEXT | Failure reason if FAILED |

### `eval_runs` (Phase 6)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `eval_job_id` | UUID | FK → `eval_jobs.id` |
| `plan_name` | VARCHAR(255) | — |
| `plan_version` | INTEGER | — |
| `metrics` | JSONB | `{"recall@5": 0.72, "mrr": 0.61, "ndcg@10": 0.58, ...}` |
| `sample_count` | INTEGER | — |
| `duration_seconds` | FLOAT | — |
| `completed_at` | TIMESTAMPTZ | — |

### `cost_entries` (Phase 7)

Hourly aggregation of `usage_records`.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `plan_name` | VARCHAR(255) | — |
| `plan_version` | INTEGER | — |
| `window_start` | TIMESTAMPTZ | Truncated to hour |
| `query_count` | INTEGER | — |
| `cache_hit_count` | INTEGER | — |
| `total_chars` | BIGINT | Sum of query_chars |
| `estimated_cost_usd` | FLOAT | Computed from model_pricing |
| `created_at` | TIMESTAMPTZ | When this aggregate was computed |

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
versions: Mapped[list[PlanVersion]] = relationship(
    "PlanVersion",
    back_populates="plan",
    lazy="selectin",
    order_by="PlanVersion.version",
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
