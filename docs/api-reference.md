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

**Path parameter:** `plan_name` — name of the retrieval plan (lowercase slug).

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
| `metadata_filters` | object | No | Key-value filters merged over the plan's `metadata_filters`. Request-level values override plan-level values for matching keys. |

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
| `plan_name` | string | Name of the plan that served the query. |
| `version` | int | Plan version number used (the current deployed version). |
| `cache_hit` | bool | Whether the result was served from the semantic query cache. |
| `result_count` | int | Number of chunks returned. |
| `results[].id` | string | The vector ID from Qdrant (payload field `id` if present, else Qdrant point ID). |
| `results[].score` | float | Cosine / dot / Euclidean similarity score from the index. Higher = more similar. |
| `results[].text` | string | Text content from the `text` payload field of the Qdrant point. |
| `results[].metadata` | object | All other payload fields from the Qdrant point. |

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PLAN_NOT_FOUND` | No plan with this name, or plan is archived. |
| 503 | `EMBEDDING_PROVIDER_ERROR` | Embedding model unavailable or not installed. |
| 503 | `INDEX_BACKEND_ERROR` | Qdrant unreachable or collection not found. |

---

## Plans

Plans are immutable versioned retrieval configurations. Every change to embedding or index config creates a new version. The current version is always the latest.

### `POST /v1/plans`

Create a new plan with version 1.

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
    "top_k": 20,
    "rerank_top_k": 5,
    "reranker": null,
    "hybrid_alpha": null,
    "metadata_filters": null,
    "tenant_isolation_field": null,
    "cache_enabled": true,
    "cache_ttl_seconds": 3600,
    "max_tokens_per_query": null,
    "change_comment": "initial version"
  }
}
```

**Plan name rules:** Lowercase letters, digits, and hyphens. Cannot start or end with a hyphen. Max 255 chars. Must be globally unique.

**Config field reference:**

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
| `quantization` | string\|null | null | One of: `scalar`, `product`, null | Vector quantization. Applied at index creation time, not at query time. |
| `top_k` | int | 10 | `>= 1` | Number of candidates returned from the ANN index. |
| `rerank_top_k` | int\|null | null | `>= 1` and `<= top_k` | Number of results returned after reranking. Requires `reranker`. |
| `reranker` | string\|null | null | — | Reranker model identifier. Phase 8. |
| `hybrid_alpha` | float\|null | null | `0.0–1.0` | Dense/sparse blend weight. `1.0` = pure dense. `0.0` = pure sparse. |
| `metadata_filters` | object\|null | null | — | Default metadata filter applied to every query against this plan. |
| `tenant_isolation_field` | string\|null | null | — | Payload field used for tenant-scoped filtering. |
| `cache_enabled` | bool | true | — | Enable semantic query cache for this plan. |
| `cache_ttl_seconds` | int | 3600 | `>= 0` | Cache TTL. `0` = disabled. |
| `max_tokens_per_query` | int\|null | null | `> 0` | Token limit per query (context packing, Phase 8). |
| `change_comment` | string | `""` | Max 2048 chars | Human-readable note about this version's changes. |

**Provider-modality compatibility matrix:**

| Provider | Supported modalities | Notes |
|---|---|---|
| `sentence_transformers` | `text` | Runs in-process via threadpool. |
| `openai` | `text` | Calls OpenAI Embeddings API. Requires `OPENAI_API_KEY`. |
| `clip` | `text`, `image` | Both modalities in one model. Phase 8. |
| `whisper` | `audio` | Transcription → text → embed pipeline. Phase 8. |
| `video_frame` | `video` | Frame sampling + ViT mean-pool. Phase 8. |

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
  "current_version": {
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
    "top_k": 20,
    "rerank_top_k": 5,
    "reranker": null,
    "hybrid_alpha": null,
    "metadata_filters": null,
    "tenant_isolation_field": null,
    "cache_enabled": true,
    "cache_ttl_seconds": 3600,
    "max_tokens_per_query": null,
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
| 409 | `CONFLICT` | A plan named `{name}` already exists. |
| 422 | `VALIDATION_ERROR` | Config validation failed. `detail.errors` contains all failures. |

---

### `GET /v1/plans`

List plans, newest first. Cursor-paginated.

**Query parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `cursor` | string | — | Opaque pagination cursor from previous response. |
| `limit` | int | 20 | Results per page. Range: 1–100. |
| `include_archived` | bool | false | Include archived (soft-deleted) plans. |

**Response 200:**

```json
{
  "items": [ /* PlanResponse objects */ ],
  "total": 47,
  "cursor": "MTU=",
  "has_more": true
}
```

Pass `cursor` from the response as the `cursor` query param in the next request. When `has_more` is false, you have all results.

---

### `GET /v1/plans/{name}`

Get a plan and its current version.

**Response 200:** `PlanResponse` (same shape as POST response).

**Error:** 404 `PLAN_NOT_FOUND`.

---

### `POST /v1/plans/{name}/versions`

Create a new version of an existing plan. The new version immediately becomes current.

**Request body:**

```json
{
  "created_by": "alice",
  "config": { /* PlanVersionConfig — same fields as above */ }
}
```

**Response 201:** `PlanVersionResponse`

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PLAN_NOT_FOUND` | No plan with this name. |
| 409 | `CONFLICT` | Plan is archived. |
| 409 | `DUPLICATE_CONFIG_HASH` | An existing version has identical retrieval config. `detail.existing_version` gives the version number. |
| 422 | `VALIDATION_ERROR` | Config validation failed. |

---

### `GET /v1/plans/{name}/versions`

List all versions of a plan, oldest first.

**Response 200:** `list[PlanVersionResponse]`

---

### `GET /v1/plans/{name}/versions/{version_num}`

Get a specific version.

**Response 200:** `PlanVersionResponse`

**Error:** 404 `PLAN_VERSION_NOT_FOUND`.

---

### `POST /v1/plans/{name}/clone`

Clone a plan's current version as version 1 of a new plan. The two plans share no state after creation — changes to one do not affect the other.

**Request body:**

```json
{
  "new_name": "my-docs-experiment",
  "description": "Clone for A/B test",
  "created_by": "alice"
}
```

**Response 201:** `PlanResponse` for the new plan.

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 404 | `PLAN_NOT_FOUND` | Source plan not found or has no current version. |
| 409 | `CONFLICT` | `new_name` already exists. |

---

### `DELETE /v1/plans/{name}`

Soft-delete (archive) a plan. Archived plans cannot receive new versions or new deployments. Existing data is preserved.

**Response:** 204 No Content.

---

## Deployments

Deployments bind a plan version to live traffic. A deployment goes through a state machine from `PENDING` → `ACTIVE` (or rolls back).

### `POST /v1/plans/{plan_name}/deployments`

Deploy a version of a plan.

**Instant deployment** — version goes live immediately at 100% traffic:

```json
{
  "plan_version": 2,
  "change_note": "upgrading to bge-m3 from bge-base",
  "created_by": "alice"
}
```

**Gradual deployment** — traffic is incremented by `rollout_step_percent` every `rollout_step_interval_seconds`:

```json
{
  "plan_version": 2,
  "rollout_step_percent": 10.0,
  "rollout_step_interval_seconds": 300,
  "rollback_recall_threshold": 0.80,
  "rollback_error_rate_threshold": 0.02,
  "change_note": "gradual rollout — 10% every 5 min",
  "created_by": "alice"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `plan_version` | int | Yes | Version number to deploy. Must exist. |
| `rollout_step_percent` | float | Both or neither | Traffic increase per step (1–100). |
| `rollout_step_interval_seconds` | int | Both or neither | Seconds between steps. Min 10. |
| `rollback_recall_threshold` | float | No | Auto-rollback if Recall@10 drops below this value (0.0–1.0). |
| `rollback_error_rate_threshold` | float | No | Auto-rollback if error rate exceeds this value (0.0–1.0). |
| `change_note` | string | No | Human-readable description. Max 2048 chars. |
| `created_by` | string | Yes | Identity of deployer. |

**Response 201:** `DeploymentResponse`

```json
{
  "id": "018e7a2c-0000-7000-c345-678901abcdef",
  "plan_name": "my-docs",
  "plan_version": 2,
  "status": "ACTIVE",
  "traffic_weight": 1.0,
  "rollout_step_percent": null,
  "rollout_step_interval_seconds": null,
  "rollback_recall_threshold": null,
  "rollback_error_rate_threshold": null,
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
| 404 | `PLAN_NOT_FOUND` | Plan not found or archived. |
| 404 | `PLAN_VERSION_NOT_FOUND` | Specified version doesn't exist. |
| 409 | `CONFLICT` | Plan already has a live deployment (ACTIVE or ROLLING_OUT). |
| 422 | `VALIDATION_ERROR` | Only one of rollout_step_percent / rollout_step_interval_seconds provided. |

---

### `GET /v1/plans/{plan_name}/deployments`

List all deployments for a plan, newest first (up to 20).

**Response 200:** `list[DeploymentResponse]`

---

### `GET /v1/plans/{plan_name}/deployments/{deployment_id}`

Get a specific deployment.

**Response 200:** `DeploymentResponse`

**Error:** 404 `DEPLOYMENT_NOT_FOUND`.

---

### `POST /v1/plans/{plan_name}/deployments/{deployment_id}/rollback`

Immediately roll back a live deployment. Sets traffic_weight to 0 and status to `ROLLED_BACK`. Clears the Redis active-deployment key so subsequent queries fall back to Postgres for plan config resolution.

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
| 404 | `DEPLOYMENT_NOT_FOUND` | Deployment not found or belongs to a different plan. |
| 409 | `DEPLOYMENT_STATE_ERROR` | Deployment is not live (can only roll back ACTIVE or ROLLING_OUT). |

---

## Pagination

List endpoints use opaque cursor pagination:

```
GET /v1/plans?limit=10
→ { "items": [...], "total": 47, "cursor": "MTB=", "has_more": true }

GET /v1/plans?cursor=MTB=&limit=10
→ { "items": [...], "total": 47, "cursor": "MjA=", "has_more": true }

GET /v1/plans?cursor=MjA=&limit=10
→ { "items": [...], "total": 47, "cursor": null, "has_more": false }
```

The cursor is an opaque base64 string. Do not parse it — only pass it back as-is.

---

## Common Patterns

### Create a plan and deploy it

```bash
# 1. Create the plan
curl -X POST http://localhost:8000/v1/plans \
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
      "distance_metric": "cosine",
      "top_k": 20,
      "cache_enabled": true,
      "cache_ttl_seconds": 1800
    }
  }'

# 2. Deploy it
curl -X POST http://localhost:8000/v1/plans/product-search/deployments \
  -H 'Content-Type: application/json' \
  -d '{"plan_version": 1, "created_by": "eng-team"}'

# 3. Query it
curl -X POST http://localhost:8000/v1/query/product-search \
  -H 'Content-Type: application/json' \
  -d '{"query": "red running shoes size 10"}'
```

### Gradual rollout with guard rails

```bash
curl -X POST http://localhost:8000/v1/plans/product-search/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "plan_version": 2,
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
  http://localhost:8000/v1/plans/product-search/deployments/{deployment_id}/rollback \
  -H 'Content-Type: application/json' \
  -d '{"reason": "latency spike detected", "created_by": "oncall"}'
```

### Iterate on config without changing identity

```bash
# Create a new version (name stays the same, version number increments)
curl -X POST http://localhost:8000/v1/plans/product-search/versions \
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
      "top_k": 30,
      "cache_enabled": true,
      "cache_ttl_seconds": 1800,
      "change_comment": "increased top_k from 20 to 30 for reranker headroom"
    }
  }'
```
