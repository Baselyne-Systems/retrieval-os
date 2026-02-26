# API Reference

Base URL: `http://localhost:8000`

All endpoints accept and return `application/json` unless noted. Timestamps are ISO 8601 with timezone (`2026-02-25T12:00:00Z`). IDs are UUIDv7 strings.

Error responses always follow this envelope:

```json
{
  "error": "MACHINE_READABLE_CODE",
  "message": "Human-readable description",
  "detail": { "errors": ["field-level details if applicable"] }
}
```

---

## Infrastructure

### `GET /health`

Always returns 200. Use for liveness probes.

```json
{ "status": "ok" }
```

---

### `GET /ready`

Checks all critical dependencies. Returns 200 when all healthy, 503 when any are down.

```json
{
  "status": "ok",
  "checks": {
    "postgres": "ok",
    "redis": "ok"
  }
}
```

503 response:
```json
{
  "status": "degraded",
  "checks": {
    "postgres": "error",
    "redis": "ok"
  }
}
```

---

### `GET /metrics`

Prometheus text format metrics. Scraped by Prometheus at 15s intervals.

```
Content-Type: text/plain; version=0.0.4; charset=utf-8
```

---

### `GET /v1/info`

```json
{
  "service": "retrieval-os-api",
  "version": "0.1.0",
  "environment": "development"
}
```

---

## Serving

### `POST /v1/query/{plan_name}`

Execute a retrieval query. This is the hot path — P99 target < 200ms.

**Path parameter:** `plan_name` — name of the project (lowercase slug).

**Request body:**

```json
{
  "query": "what is retrieval augmented generation?",
  "metadata_filters": {
    "language": "en",
    "doc_type": "article"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Query text. Min 1 char, max 8192 chars. |
| `metadata_filters` | object | No | Key-value filters merged over the deployment's `metadata_filters`. Request-level values override deployment-level values for matching keys. |

**Response 200:**

```json
{
  "plan_name": "my-docs",
  "version": 3,
  "cache_hit": false,
  "result_count": 10,
  "results": [
    {
      "id": "chunk-abc123",
      "score": 0.923,
      "text": "Retrieval augmented generation (RAG) is a technique...",
      "metadata": {
        "source": "s3://my-bucket/docs/rag-intro.pdf",
        "page": 1,
        "language": "en"
      }
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `plan_name` | string | Name of the project that served the query. |
| `version` | int | Index config version number used (the active deployment's index_config_version). |
| `cache_hit` | bool | Whether the result was served from the semantic query cache. |
| `result_count` | int | Number of chunks returned. |
| `results[].id` | string | The vector ID from Qdrant (payload field `id` if present, else Qdrant point ID). |
| `results[].score` | float | Cosine / dot / Euclidean similarity score from the index. Higher = more similar. |
| `results[].text` | string | Text content from the `text` payload field of the Qdrant point. |
| `results[].metadata` | object | All other payload fields from the Qdrant point. |

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PROJECT_NOT_FOUND` | No project with this name, or project is archived. |
| 503 | `EMBEDDING_PROVIDER_ERROR` | Embedding model unavailable or not installed. |
| 503 | `INDEX_BACKEND_ERROR` | Qdrant unreachable or collection not found. |
| 504 | `QUERY_TIMEOUT` | The full retrieval pipeline (embed + ANN + cache write) exceeded `QUERY_TIMEOUT_SECONDS` (default 30 s). |

---

## Projects

Projects are the named containers for retrieval use cases. Each project has versioned IndexConfigs (build-time config) and Deployments (search config + lifecycle).

### `POST /v1/projects`

Create a new project with its first IndexConfig (version 1).

**Request body:**

```json
{
  "name": "my-docs",
  "description": "Documentation search for the public site",
  "created_by": "alice",
  "config": {
    "embedding_provider": "sentence_transformers",
    "embedding_model": "BAAI/bge-m3",
    "embedding_dimensions": 1024,
    "modalities": ["text"],
    "embedding_batch_size": 32,
    "embedding_normalize": true,
    "index_backend": "qdrant",
    "index_collection": "my_docs_v1",
    "distance_metric": "cosine",
    "quantization": null,
    "change_comment": "initial version"
  }
}
```

**Project name rules:** Lowercase letters, digits, and hyphens. Cannot start or end with a hyphen. Max 255 chars. Must be globally unique.

**IndexConfig field reference (build-time only — search config goes on Deployment):**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `embedding_provider` | string | — | Required. One of: `sentence_transformers`, `openai`, `clip`, `whisper`, `video_frame` | Embedding model provider. |
| `embedding_model` | string | — | Required | Provider-specific model identifier (e.g. `BAAI/bge-m3`, `text-embedding-3-large`). |
| `embedding_dimensions` | int | 768 | `> 0` | Output vector dimensionality. Must match the collection's configured dimensions. |
| `modalities` | list[string] | `["text"]` | Non-empty. Must be compatible with provider. | Active modalities. See provider-modality matrix below. |
| `embedding_batch_size` | int | 32 | `> 0` | Texts per embed call. Larger = faster throughput, more memory. |
| `embedding_normalize` | bool | true | — | L2-normalise output vectors. Required for cosine distance. |
| `index_backend` | string | `"qdrant"` | One of: `qdrant`, `pgvector` | Vector index backend. |
| `index_collection` | string | — | Required | Collection name in Qdrant (or table name in pgvector). |
| `distance_metric` | string | `"cosine"` | One of: `cosine`, `dot`, `euclidean` | Must match the collection's configured metric. |
| `quantization` | string\|null | null | One of: `scalar`, `product`, null | Vector quantization. Applied at index creation time. |
| `change_comment` | string | `""` | Max 2048 chars | Human-readable note about this version's changes. |

**Provider-modality compatibility matrix:**

| Provider | Supported modalities | Notes |
|---|---|---|
| `sentence_transformers` | `text` | Runs in-process via threadpool. |
| `openai` | `text` | Calls OpenAI Embeddings API. Requires `OPENAI_API_KEY`. |
| `clip` | `text`, `image` | Both modalities in one model. |
| `whisper` | `audio` | Transcription → text → embed pipeline. |
| `video_frame` | `video` | Frame sampling + ViT mean-pool. |

**Response 201:**

```json
{
  "id": "018e7a2b-3f4c-7000-a123-456789abcdef",
  "name": "my-docs",
  "description": "Documentation search for the public site",
  "is_archived": false,
  "created_at": "2026-02-25T12:00:00Z",
  "updated_at": "2026-02-25T12:00:00Z",
  "created_by": "alice",
  "current_index_config": {
    "id": "018e7a2b-4000-7000-b234-567890abcdef",
    "version": 1,
    "is_current": true,
    "embedding_provider": "sentence_transformers",
    "embedding_model": "BAAI/bge-m3",
    "embedding_dimensions": 1024,
    "modalities": ["text"],
    "embedding_batch_size": 32,
    "embedding_normalize": true,
    "index_backend": "qdrant",
    "index_collection": "my_docs_v1",
    "distance_metric": "cosine",
    "quantization": null,
    "change_comment": "initial version",
    "config_hash": "a3b4c5d6e7f8...",
    "created_at": "2026-02-25T12:00:00Z",
    "created_by": "alice"
  }
}
```

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 409 | `CONFLICT` | A project named `{name}` already exists. |
| 422 | `VALIDATION_ERROR` | Config validation failed. `detail.errors` contains all failures. |

---

### `GET /v1/projects`

List projects, newest first. Cursor-paginated.

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `cursor` | string | — | Opaque pagination cursor from previous response. |
| `limit` | int | 20 | Results per page. Range: 1–100. |
| `include_archived` | bool | false | Include archived (soft-deleted) projects. |

**Response 200:**

```json
{
  "items": [ /* ProjectResponse objects */ ],
  "total": 47,
  "cursor": "MTU=",
  "has_more": true
}
```

Pass `cursor` from the response as the `cursor` query param in the next request. When `has_more` is false, you have all results.

---

### `GET /v1/projects/{name}`

Get a project and its current IndexConfig.

**Response 200:** `ProjectResponse` (same shape as POST response).

**Error:** 404 `PROJECT_NOT_FOUND`.

---

### `POST /v1/projects/{name}/index-configs`

Create a new IndexConfig version. The new version immediately becomes current.

**Request body:**

```json
{
  "created_by": "alice",
  "config": { /* IndexConfigInput — same fields as above */ }
}
```

**Response 201:** `IndexConfigResponse`

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PROJECT_NOT_FOUND` | No project with this name. |
| 409 | `CONFLICT` | Project is archived. |
| 409 | `DUPLICATE_CONFIG_HASH` | An existing version has identical index config. `detail.existing_version` gives the version number. |
| 422 | `VALIDATION_ERROR` | Config validation failed. |

---

### `GET /v1/projects/{name}/index-configs`

List all IndexConfig versions for a project, oldest first.

**Response 200:** `list[IndexConfigResponse]`

---

### `GET /v1/projects/{name}/index-configs/{version_num}`

Get a specific IndexConfig version.

**Response 200:** `IndexConfigResponse`

**Error:** 404 `INDEX_CONFIG_NOT_FOUND`.

---

### `POST /v1/projects/{name}/clone`

Clone a project's current IndexConfig as version 1 of a new project. The two projects share no state after creation.

**Request body:**

```json
{
  "new_name": "my-docs-experiment",
  "description": "Clone for A/B test",
  "created_by": "alice"
}
```

**Response 201:** `ProjectResponse` for the new project.

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PROJECT_NOT_FOUND` | Source project not found or has no current IndexConfig. |
| 409 | `CONFLICT` | `new_name` already exists. |

---

### `DELETE /v1/projects/{name}`

Soft-delete (archive) a project. Archived projects cannot receive new index configs or deployments. Existing data is preserved.

**Response:** 204 No Content.

---

## Deployments

Deployments bind an IndexConfig version to live traffic and carry the runtime search config. A deployment goes through a state machine from `PENDING` → `ACTIVE` (or rolls back).

### `POST /v1/projects/{project_name}/deployments`

Deploy an IndexConfig version with search config.

**Instant deployment** — goes live immediately at 100% traffic:

```json
{
  "index_config_version": 2,
  "top_k": 10,
  "reranker": "cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2",
  "rerank_top_k": 5,
  "cache_enabled": true,
  "cache_ttl_seconds": 3600,
  "change_note": "upgrading to bge-m3, adding cross-encoder",
  "created_by": "alice"
}
```

**Gradual deployment** — traffic is incremented by `rollout_step_percent` every `rollout_step_interval_seconds`:

```json
{
  "index_config_version": 2,
  "top_k": 10,
  "rollout_step_percent": 10.0,
  "rollout_step_interval_seconds": 300,
  "rollback_recall_threshold": 0.80,
  "rollback_error_rate_threshold": 0.02,
  "change_note": "gradual rollout — 10% every 5 min",
  "created_by": "alice"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `index_config_version` | int | Yes | — | IndexConfig version number to deploy. Must exist. |
| `top_k` | int | No | 10 | Number of ANN candidates returned from the index. |
| `reranker` | string\|null | No | null | Reranker in `"provider:model"` format. Supported: `cross_encoder`, `cohere`. |
| `rerank_top_k` | int\|null | No | null | Post-rerank result count. Must be ≤ `top_k`. |
| `hybrid_alpha` | float\|null | No | null | Dense/sparse blend weight (0.0–1.0). |
| `metadata_filters` | object\|null | No | null | Default metadata filter for all queries. |
| `tenant_isolation_field` | string\|null | No | null | Payload field for tenant-scoped filtering. |
| `cache_enabled` | bool | No | true | Enable semantic query cache. |
| `cache_ttl_seconds` | int | No | 3600 | Cache TTL. `0` = disabled. |
| `max_tokens_per_query` | int\|null | No | null | Token cap for context packing. |
| `rollout_step_percent` | float | Both or neither | — | Traffic increase per step (1–100). |
| `rollout_step_interval_seconds` | int | Both or neither | — | Seconds between steps. Min 10. |
| `rollback_recall_threshold` | float | No | null | Auto-rollback if Recall@5 drops below this (0.0–1.0). |
| `rollback_error_rate_threshold` | float | No | null | Auto-rollback if error rate exceeds this (0.0–1.0). |
| `eval_dataset_uri` | string\|null | No | null | S3 URI of a JSONL eval dataset. If set, an eval job is automatically queued when this deployment is activated. Format: `s3://bucket/path.jsonl`. |
| `change_note` | string | No | `""` | Human-readable description. Max 2048 chars. |
| `created_by` | string | Yes | — | Identity of deployer. |

**Response 201:** `DeploymentResponse`

```json
{
  "id": "018e7a2c-0000-7000-c345-678901abcdef",
  "plan_name": "my-docs",
  "index_config_version": 2,
  "status": "ACTIVE",
  "traffic_weight": 1.0,
  "top_k": 10,
  "reranker": null,
  "rerank_top_k": null,
  "hybrid_alpha": null,
  "metadata_filters": null,
  "cache_enabled": true,
  "cache_ttl_seconds": 3600,
  "rollout_step_percent": null,
  "rollout_step_interval_seconds": null,
  "rollback_recall_threshold": null,
  "rollback_error_rate_threshold": null,
  "eval_dataset_uri": null,
  "change_note": "upgrading to bge-m3",
  "created_at": "2026-02-25T12:05:00Z",
  "updated_at": "2026-02-25T12:05:00Z",
  "created_by": "alice",
  "activated_at": "2026-02-25T12:05:00Z",
  "rolled_back_at": null,
  "rollback_reason": null
}
```

**Deployment status values:**

| Status | Meaning |
|---|---|
| `PENDING` | Created; not yet live. |
| `ROLLING_OUT` | Live, with traffic weight < 1.0. Rollout stepper will advance it. |
| `ACTIVE` | Fully live at 100% traffic. |
| `ROLLING_BACK` | Rollback in progress (transitions to ROLLED_BACK immediately). |
| `ROLLED_BACK` | Traffic weight set to 0. Rollback reason recorded. |
| `FAILED` | Unrecoverable error during deployment. |

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PROJECT_NOT_FOUND` | Project not found or archived. |
| 404 | `INDEX_CONFIG_NOT_FOUND` | Specified IndexConfig version doesn't exist. |
| 409 | `CONFLICT` | Project already has a live deployment (ACTIVE or ROLLING_OUT). |
| 422 | `VALIDATION_ERROR` | Only one of rollout_step_percent / rollout_step_interval_seconds provided. |

---

### `GET /v1/projects/{project_name}/deployments`

List all deployments for a project, newest first (up to 20).

**Response 200:** `list[DeploymentResponse]`

---

### `GET /v1/projects/{project_name}/deployments/{deployment_id}`

Get a specific deployment.

**Response 200:** `DeploymentResponse`

**Error:** 404 `DEPLOYMENT_NOT_FOUND`.

---

### `POST /v1/projects/{project_name}/deployments/{deployment_id}/rollback`

Immediately roll back a live deployment. Sets traffic_weight to 0 and status to `ROLLED_BACK`. Clears the Redis serving config key so subsequent queries fall back to Postgres for config resolution.

**Request body:**

```json
{
  "reason": "recall@10 dropped from 0.84 to 0.71 after deploy",
  "created_by": "oncall-alice"
}
```

**Response 200:** `DeploymentResponse` with updated status.

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `DEPLOYMENT_NOT_FOUND` | Deployment not found or belongs to a different project. |
| 409 | `DEPLOYMENT_STATE_ERROR` | Deployment is not live (can only roll back ACTIVE or ROLLING_OUT). |

---

## Webhooks

Webhooks deliver signed event notifications to your HTTP endpoints when key state changes occur in the system.

### `POST /v1/webhooks`

Register a new subscription.

**Request body:**

```json
{
  "url": "https://your-service.example.com/hooks/retrieval-os",
  "events": ["deployment.status_changed", "eval.regression_detected"],
  "secret": "my-shared-secret",
  "description": "Slack alerting hook"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `url` | string | Yes | HTTPS URL to deliver events to. Must be a valid HTTP(S) URL. |
| `events` | list[string] | No | Event types to subscribe to. Empty list (default) = all events. |
| `secret` | string | No | Shared secret for HMAC-SHA256 request signing. |
| `description` | string | No | Human-readable label. Max 1024 chars. |

**Available event types:**

| Event | Trigger |
|---|---|
| `deployment.status_changed` | Any deployment status transition (ACTIVE, ROLLED_BACK, etc.). |
| `eval.regression_detected` | An eval job completes and finds recall degraded vs. the previous run. |
| `project.index_config_created` | A new IndexConfig version is created under a project. |
| `cost.threshold_exceeded` | Cost intelligence detects spending above a configured threshold. |

**Response 201:**

```json
{
  "id": "018e7a2c-1111-7000-d456-789012abcdef",
  "url": "https://your-service.example.com/hooks/retrieval-os",
  "events": ["deployment.status_changed", "eval.regression_detected"],
  "description": "Slack alerting hook",
  "is_active": true,
  "created_at": "2026-02-25T12:00:00Z",
  "updated_at": "2026-02-25T12:00:00Z"
}
```

Note: `secret` is never returned in responses.

---

### `GET /v1/webhooks`

List all subscriptions.

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `offset` | int | 0 | Number of results to skip. |
| `limit` | int | 50 | Results per page. Max 200. |

**Response 200:**

```json
{
  "items": [ /* WebhookSubscriptionResponse objects */ ],
  "total": 3
}
```

---

### `GET /v1/webhooks/{webhook_id}`

Get a single subscription.

**Response 200:** `WebhookSubscriptionResponse`

**Error:** 404 if not found.

---

### `DELETE /v1/webhooks/{webhook_id}`

Delete a subscription. Events will no longer be delivered to its URL.

**Response:** 204 No Content.

**Error:** 404 if not found.

---

### Webhook payload format

Every outbound request is a `POST` with `Content-Type: application/json`:

```json
{
  "event": "deployment.status_changed",
  "timestamp": "2026-02-25T12:05:00Z",
  "data": {
    "deployment_id": "018e7a2c-0000-7000-c345-678901abcdef",
    "plan_name": "my-docs",
    "status": "ROLLED_BACK",
    "reason": "recall@5 0.5100 < threshold 0.7500",
    "triggered_by": "watchdog"
  }
}
```

### HMAC-SHA256 signature verification

When a `secret` is configured, the request includes:

```
X-Retrieval-OS-Signature: sha256=<hex_digest>
```

The digest is computed over the raw JSON request body:

```python
import hashlib, hmac

def verify(secret: str, body: bytes, header: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header)
```

### Delivery behaviour

- Delivery is fire-and-forget — the sending request is never blocked.
- Up to 3 attempts are made per event per subscription.
- Retries use exponential back-off: 1s, 2s, 4s.
- HTTP 5xx responses and network errors trigger a retry. 4xx responses do not.
- Failed deliveries after all retries are silently dropped (check your endpoint logs).

---

## Ingestion

The ingestion API accepts documents, chunks them by word boundary, embeds each chunk, and upserts the vectors into the project's Qdrant collection. All work is performed asynchronously by a background job runner.

### `POST /v1/projects/{project_name}/ingest`

Submit a batch ingestion job. Returns immediately with status `QUEUED`.

**Inline documents:**

```json
{
  "index_config_version": 1,
  "documents": [
    {
      "id": "doc-001",
      "content": "Retrieval augmented generation (RAG) is a technique...",
      "metadata": {"source": "rag-intro.pdf", "page": 1}
    },
    {
      "id": "doc-002",
      "content": "Vector databases store high-dimensional embeddings...",
      "metadata": {"source": "vector-db-primer.pdf", "page": 1}
    }
  ],
  "chunk_size": 512,
  "overlap": 64,
  "created_by": "alice"
}
```

**S3 JSONL source:**

```json
{
  "index_config_version": 1,
  "source_uri": "s3://my-bucket/datasets/docs-v1.jsonl",
  "chunk_size": 256,
  "overlap": 32,
  "created_by": "pipeline-bot"
}
```

Each line in the JSONL file must be: `{"id": "...", "content": "...", "metadata": {}}`.

| Field | Type | Required | Description |
|---|---|---|---|
| `index_config_version` | int | Yes | IndexConfig version whose Qdrant collection to populate. Must exist. |
| `documents` | list | One of | Inline document list. Mutually exclusive with `source_uri`. |
| `source_uri` | string | One of | S3 URI of a JSONL file. Mutually exclusive with `documents`. |
| `chunk_size` | int | No | Max words per chunk. Range: 16–4096. Default: 512. |
| `overlap` | int | No | Word overlap between consecutive chunks. Must be < `chunk_size`. Default: 64. |
| `created_by` | string | No | Identity of submitter. |

**Response 202:**

```json
{
  "id": "018e7a2c-2222-7000-e567-890123abcdef",
  "plan_name": "my-docs",
  "index_config_id": "018e7a2b-4000-7000-b234-567890abcdef",
  "index_config_version": 1,
  "source_uri": null,
  "chunk_size": 512,
  "overlap": 64,
  "status": "QUEUED",
  "total_docs": null,
  "total_chunks": null,
  "indexed_chunks": null,
  "failed_chunks": null,
  "error_message": null,
  "duplicate_of": null,
  "created_at": "2026-02-25T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "created_by": "alice"
}
```

| Field | Type | Description |
|---|---|---|
| `plan_name` | string | Name of the project this job belongs to. |
| `index_config_id` | UUID | ID of the IndexConfig used for embedding and collection selection. |
| `index_config_version` | int | Human-readable version number of the IndexConfig. |
| `status` | string | `QUEUED`, `RUNNING`, `COMPLETED`, or `FAILED`. |
| `total_docs` | int\|null | Total documents processed. Populated when COMPLETED. |
| `total_chunks` | int\|null | Total chunks generated across all documents. |
| `indexed_chunks` | int\|null | Chunks successfully upserted to Qdrant. |
| `failed_chunks` | int\|null | Chunks that failed to embed or upsert. |
| `error_message` | string\|null | Set if the job reaches FAILED status. |
| `duplicate_of` | string\|null | If set, this job was a dedup no-op; the value is the `id` of the prior COMPLETED job for the same `(project_name, index_config_version)`. No embedding or Qdrant upsert was performed. |
| `started_at` | timestamp\|null | When the background runner claimed the job. |
| `completed_at` | timestamp\|null | When the job reached COMPLETED or FAILED. |

**Error responses:**

| Status | Cause |
|---|---|
| 404 | Project not found or archived. |
| 404 | Specified `index_config_version` not found. |
| 422 | Both or neither of `documents` / `source_uri` provided, or `overlap >= chunk_size`. |

---

### `GET /v1/projects/{project_name}/ingest`

List ingestion jobs for a project, newest first.

**Query parameters:** `offset` (default 0), `limit` (default 50, max 200).

**Response 200:**

```json
{
  "items": [ /* IngestionJobResponse objects */ ],
  "total": 12
}
```

---

### `GET /v1/projects/{project_name}/ingest/{job_id}`

Get a single ingestion job.

**Response 200:** `IngestionJobResponse`

**Error:** 404 if not found or if the job belongs to a different project.

---

### Chunking behaviour

Documents are split on word boundaries (whitespace-delimited tokens):

- A document with ≤ `chunk_size` words produces a single chunk.
- Longer documents produce overlapping windows: chunk N starts `chunk_size - overlap` words after chunk N-1.
- Each chunk inherits the document's `metadata` plus a `chunk_index` field.
- Chunks are embedded in batches using the IndexConfig's embedding settings.
- The Qdrant collection is auto-created on the first ingestion if it doesn't exist.

---

## Pagination

List endpoints use opaque cursor pagination:

```
GET /v1/projects?limit=10
→ { "items": [...], "total": 47, "cursor": "MTB=", "has_more": true }

GET /v1/projects?cursor=MTB=&limit=10
→ { "items": [...], "total": 47, "cursor": "MjA=", "has_more": true }

GET /v1/projects?cursor=MjA=&limit=10
→ { "items": [...], "total": 47, "cursor": null, "has_more": false }
```

The cursor is an opaque base64 string. Do not parse it — only pass it back as-is.

---

## Common Patterns

### Create a project and deploy it

```bash
# 1. Create the project with its first IndexConfig (index-only fields)
curl -X POST http://localhost:8000/v1/projects \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "product-search",
    "description": "E-commerce product retrieval",
    "created_by": "eng-team",
    "config": {
      "embedding_provider": "sentence_transformers",
      "embedding_model": "BAAI/bge-m3",
      "embedding_dimensions": 1024,
      "modalities": ["text"],
      "index_backend": "qdrant",
      "index_collection": "products_v1",
      "distance_metric": "cosine"
    }
  }'

# 2. Deploy IndexConfig v1 with search config
curl -X POST http://localhost:8000/v1/projects/product-search/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "index_config_version": 1,
    "top_k": 20,
    "cache_enabled": true,
    "cache_ttl_seconds": 1800,
    "created_by": "eng-team"
  }'

# 3. Query it
curl -X POST http://localhost:8000/v1/query/product-search \
  -H 'Content-Type: application/json' \
  -d '{"query": "red running shoes size 10"}'
```

### Gradual rollout with guard rails

```bash
curl -X POST http://localhost:8000/v1/projects/product-search/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "index_config_version": 2,
    "top_k": 20,
    "rollout_step_percent": 10.0,
    "rollout_step_interval_seconds": 300,
    "rollback_recall_threshold": 0.78,
    "rollback_error_rate_threshold": 0.01,
    "change_note": "testing new embedding model — 10% increments over 50 min",
    "created_by": "alice"
  }'
```

### Roll back immediately

```bash
curl -X POST \
  http://localhost:8000/v1/projects/product-search/deployments/{deployment_id}/rollback \
  -H 'Content-Type: application/json' \
  -d '{"reason": "latency spike detected", "created_by": "oncall"}'
```

### Iterate on index config without renaming the project

```bash
# Create a new IndexConfig version (project name stays the same, version increments)
curl -X POST http://localhost:8000/v1/projects/product-search/index-configs \
  -H 'Content-Type: application/json' \
  -d '{
    "created_by": "alice",
    "config": {
      "embedding_provider": "sentence_transformers",
      "embedding_model": "BAAI/bge-m3",
      "embedding_dimensions": 1024,
      "modalities": ["text"],
      "index_backend": "qdrant",
      "index_collection": "products_v2",
      "distance_metric": "cosine",
      "change_comment": "new collection for fresh re-index"
    }
  }'

# Then deploy v2 with updated search config (more top_k for reranker headroom)
curl -X POST http://localhost:8000/v1/projects/product-search/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "index_config_version": 2,
    "top_k": 30,
    "reranker": "cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_top_k": 5,
    "created_by": "alice"
  }'
```
