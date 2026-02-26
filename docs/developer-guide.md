# Developer Guide

---

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Runtime |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager and venv |
| Docker + Docker Compose | 24+ | Local infrastructure |
| Make | any | Task runner |
| Git | any | VCS |

**Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## First-time Setup

```bash
# Clone
git clone https://github.com/your-org/retrieval-os.git
cd retrieval-os

# Install all dependencies (including dev extras)
uv sync --all-extras

# Copy env config
cp .env.example .env

# Start infrastructure (Postgres, Redis, Qdrant, MinIO, Prometheus, Grafana, Jaeger)
make infra

# Wait ~15s for all services to be healthy, then run migrations
make migrate

# Start the API in development mode (hot reload)
make dev
```

Verify the stack is up:

```bash
curl http://localhost:8000/ready
# {"status":"ok","checks":{"postgres":"ok","redis":"ok"}}

curl http://localhost:8000/v1/info
# {"service":"retrieval-os-api","version":"0.1.0","environment":"development"}
```

---

## Make Targets

| Target | What it does |
|---|---|
| `make dev` | Start API with uvicorn `--reload` on port 8000 |
| `make infra` | `docker-compose -f docker-compose.infra.yml up -d` |
| `make stop` | Stop all Docker services |
| `make test` | Run full test suite (`uv run pytest tests/`) |
| `make lint` | Run `ruff check src tests` |
| `make fmt` | Run `ruff format src tests` and `ruff check --fix` |
| `make migrate` | Run `alembic upgrade head` |
| `make migrate-new NAME=...` | Create a new migration file |
| `make shell` | Open an IPython shell in the project venv |
| `make install` | `uv sync --all-extras` |

---

## Project Structure

```
retrieval-os/
├── alembic/                    Database migrations
│   ├── env.py                  Async migration runner
│   └── versions/
│       ├── 0001_initial_schema.py
│       ├── 0002_plans.py
│       ├── 0003_usage_records.py
│       └── 0004_deployments.py
│
├── docs/                       This documentation
│   ├── architecture.md
│   ├── api-reference.md
│   ├── configuration.md
│   ├── data-models.md
│   ├── developer-guide.md      ← you are here
│   └── observability.md
│
├── infra/
│   ├── docker/
│   │   └── Dockerfile.api
│   ├── grafana/provisioning/
│   │   ├── datasources/
│   │   └── dashboards/
│   └── prometheus/
│       ├── prometheus.yml
│       └── alert_rules/
│
├── src/retrieval_os/
│   ├── api/
│   │   ├── main.py             App factory, lifespan, middleware, exception handler
│   │   ├── background.py       Asyncio background task loops
│   │   ├── health.py           /health, /ready, /metrics, /v1/info
│   │   ├── serving_router.py   POST /v1/query/{plan_name}
│   │   └── middleware/
│   │       ├── request_id.py   UUIDv7 X-Request-ID injection
│   │       └── telemetry.py    OTel span attachment
│   │
│   ├── core/
│   │   ├── config.py           pydantic-settings root config (Settings)
│   │   ├── database.py         SQLAlchemy async engine + get_db() FastAPI dep
│   │   ├── exceptions.py       Typed exception hierarchy
│   │   ├── ids.py              UUIDv7 generator
│   │   ├── metrics.py          All Prometheus metric definitions
│   │   ├── redis_client.py     Async Redis pool + get_redis() helper
│   │   ├── s3_client.py        boto3 S3 wrapper (threadpool)
│   │   ├── telemetry.py        OTel tracer provider setup
│   │   └── schemas/
│   │       └── pagination.py   CursorPage[T] generic
│   │
│   ├── plans/
│   │   ├── models.py           Project, IndexConfig ORM
│   │   ├── validators.py       validate_plan_config(), compute_config_hash()
│   │   ├── schemas.py          Pydantic request/response schemas
│   │   ├── repository.py       DB access (SELECT FOR UPDATE on version create)
│   │   ├── service.py          Business logic: create, version, clone, archive
│   │   └── router.py           8 plan endpoints
│   │
│   ├── deployments/
│   │   ├── models.py           Deployment ORM (DeploymentStatus StrEnum)
│   │   ├── schemas.py          Request/response schemas
│   │   ├── repository.py       DB access
│   │   ├── service.py          State machine, rollout stepper, rollback watchdog
│   │   ├── traffic.py          Redis keys for active deployment + plan config
│   │   └── router.py           4 deployment endpoints
│   │
│   └── serving/
│       ├── cache.py            Redis semantic query cache
│       ├── embed_router.py     Dispatch to sentence_transformers / OpenAI
│       ├── executor.py         Full pipeline: cache → embed → ANN → cache write
│       ├── index_proxy.py      Qdrant gRPC connector
│       ├── query_router.py     Redis plan config read → executor dispatch
│       ├── schemas.py          QueryRequest, QueryResponse, ChunkResponse
│       └── usage.py            Fire-and-forget usage_record insert
│
└── tests/
    └── unit/
        ├── test_config.py
        ├── test_health.py
        ├── test_ids.py
        ├── test_plans_validators.py
        ├── test_serving.py
        └── test_deployments.py
```

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Unit tests only (no live infra needed)
uv run pytest tests/unit/ -v

# Single test file
uv run pytest tests/unit/test_plans_validators.py -v

# Single test
uv run pytest tests/unit/test_plans_validators.py::TestComputeConfigHash::test_modality_order_does_not_affect_hash -v

# With coverage
uv run pytest tests/unit/ --cov=retrieval_os --cov-report=term-missing
```

Current test structure:

| File | What it tests | Needs infra? |
|---|---|---|
| `test_ids.py` | UUIDv7 correctness (version, variant, uniqueness, ordering) | No |
| `test_config.py` | Settings defaults and env overrides | No |
| `test_health.py` | Health/ready/metrics endpoints with mocked deps | No |
| `test_plans_validators.py` | Plan config validation + config hash | No |
| `test_serving.py` | Cache key determinism, cache get/set, rerank stub, schemas | No |
| `test_deployments.py` | DeploymentStatus, is_live, schemas, Redis key format | No |

All unit tests run in ~0.1s total, no Docker required.

---

## Configuring Auto-Eval

When a deployment is activated, Retrieval-OS can automatically queue an evaluation job — no manual POST to `/v1/eval/jobs` required. Set `eval_dataset_uri` on the deployment at creation time:

```bash
curl -X POST http://localhost:8000/v1/projects/docs-search/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "index_config_version": 3,
    "top_k": 10,
    "eval_dataset_uri": "s3://my-bucket/eval/docs-search-ground-truth.jsonl",
    "rollback_recall_threshold": 0.80,
    "created_by": "deploy-pipeline"
  }'
```

**Lifecycle:**

1. Deployment created (`PENDING`).
2. Deployment activated (status → `ACTIVE`, `traffic_weight` → 1.0).
3. Auto-eval fires: a new `EvalJob` row is inserted with `created_by="system:auto-eval"` and `dataset_uri` from `eval_dataset_uri`.
4. Background `eval_job_runner` picks up the job, scores recall/MRR/NDCG against the ground-truth file.
5. Background `rollback_watchdog` reads the completed metrics. If `recall_at_5 < rollback_recall_threshold`, rollback triggers automatically.

**Ground-truth format** (one JSON object per line):

```jsonl
{"query": "how to reset password", "relevant_ids": ["doc-42", "doc-17"]}
{"query": "billing invoice download", "relevant_ids": ["doc-91"]}
```

**What happens when auto-eval fails to queue:**
`auto_queue_eval()` is best-effort — it wraps the queue call in `try/except` and logs a warning but never raises. The activation succeeds regardless. Check `retrieval_os_eval_regression_alerts_total` in Prometheus or poll `GET /v1/eval/jobs?project_name=<name>` to verify a job was created.

**Monitoring via webhooks:**
Subscribe to the `eval.job.completed` event to receive a signed webhook payload when the job finishes:

```bash
curl -X POST http://localhost:8000/v1/webhooks \
  -d '{"url": "https://your-service/hooks", "events": ["eval.job.completed", "deployment.rolled_back"]}'
```

---

## Query Timeout Tuning

Every query is wrapped in `asyncio.wait_for()` with a configurable deadline. If the Qdrant search or embedding call stalls, the server cancels the task and returns **HTTP 504** with `"error": "QUERY_TIMEOUT"` — no worker is left hanging.

**Configuration:**

```env
QUERY_TIMEOUT_SECONDS=30.0   # default; float
```

**Recommended values by use case:**

| Use case | Recommended timeout |
|---|---|
| Interactive search (UI) | 5–10 s |
| Batch pipeline (offline) | 60–120 s |
| SLA-critical API | 2–5 s |
| Development / debugging | 120 s |

**How it surfaces to clients:**

```json
HTTP 504
{
  "error": "QUERY_TIMEOUT",
  "message": "Query exceeded timeout of 30.0s",
  "request_id": "01j..."
}
```

**Interaction with circuit breaker:**
The circuit breaker wraps individual Qdrant calls. A timeout cancels the in-flight task before the circuit breaker's own error counter increments — so a single hung query will not trip the breaker. Sustained timeouts (many requests) will eventually trip it.

**Per-request override (not yet supported):**
The timeout is a global setting. If you need different SLAs per project, deploy separate Retrieval-OS instances with different `QUERY_TIMEOUT_SECONDS`.

---

## Ingestion Deduplication

If you POST `/v1/ingestion/jobs` for a project and index config version that already has a COMPLETED ingestion job, the new job will **skip all chunking, embedding, and Qdrant upserts**. It copies the stats from the original job and marks itself COMPLETED immediately.

**When it triggers:**
Same `(project_name, index_config_version)` pair, and the prior job's status is `COMPLETED`. The dedup check runs before any embedding work starts.

**What `duplicate_of` means in responses:**

```json
{
  "id": "01j...",
  "status": "COMPLETED",
  "indexed_chunks": 4231,
  "duplicate_of": "01j...<original job id>",
  ...
}
```

`duplicate_of` is the ID of the original completed job whose vectors are already in Qdrant. The duplicate job's `indexed_chunks` / `total_chunks` are copied verbatim from the original.

**How to force a re-index:**
Dedup is scoped to a specific `index_config_version`. To re-index (e.g. after updating source documents), create a new IndexConfig version first:

```bash
# Create a new index config version (increments version automatically)
curl -X POST http://localhost:8000/v1/projects/docs-search/index-configs \
  -d '{"embedding_model": "all-MiniLM-L6-v2", ..., "change_comment": "re-index with updated docs"}'

# Submit ingestion job against the new version
curl -X POST http://localhost:8000/v1/ingestion/jobs \
  -d '{"project_name": "docs-search", "index_config_version": 4, ...}'
```

---

## Running the Test Suite

The test suite has four distinct layers. Each layer requires different infrastructure and proves different guarantees.

### Unit Tests (`make test-unit`)

```bash
uv run pytest tests/unit/ -q --tb=short
```

- **Requires:** Nothing (pure Python, no Docker).
- **Run time:** ~1–2 s for 381 tests.
- **What it proves:** Correctness of individual functions — service logic, schema validators, state machines, config hash computation, cache key derivation.

### Integration Tests

```bash
uv run pytest tests/integration/ -q --tb=short
```

- **Requires:** Nothing (all I/O is mocked with `AsyncMock`).
- **What it proves:** Service-layer orchestration — that services call repositories correctly, handle errors, and return the right response shapes. All DB/Redis/Qdrant calls are replaced with `AsyncMock`.

### Microbenchmarks (`make test-benchmarks`)

```bash
uv run pytest tests/benchmarks/ -v --tb=short
```

- **Requires:** Nothing — pure Python, no infra.
- **What each file measures:**

| File | What it benchmarks | Why it matters |
|---|---|---|
| `test_iteration_velocity.py` | Throughput of chunker, config hash, RRF fusion | Sets the CPU ceiling for sustained QPS |
| `test_quality_stability.py` | Score variance across repeated RRF / rerank calls | Guards against non-determinism bugs |
| `test_operational_reliability.py` | Graceful degradation paths (no reranker, empty results) | Proves fallbacks have near-zero overhead |
| `test_cost_efficiency.py` | Aggregation loops, token estimation, recommendation logic | Ensures cost reporting adds < 1 ms/query |
| `test_latency_predictability.py` | p99 spread of JSON encode/decode, cache key derivation | Bounds the overhead budget for the hot path |

**Interpreting results:** Each test asserts a maximum elapsed time using `time.perf_counter()`. These are process-level measurements with no I/O, so they reflect pure CPU cost. A 0.05 ms JSON overhead at 10 k QPS costs ~500 ms of aggregate CPU — the thresholds are set with that budget in mind.

### E2E Failure Mode Tests (`make test-e2e`)

```bash
uv run pytest tests/e2e/ -v --tb=short
```

- **Requires:** Postgres + Redis (`make infra && make migrate`). Qdrant is **not** required.
- **Run time:** ~5–15 s.
- **What each scenario proves:**

| Scenario | What it proves |
|---|---|
| Concurrent `SELECT FOR UPDATE SKIP LOCKED` | Only one worker claims a QUEUED job at a time under concurrency |
| Redis warm cache serves stale config | Serving never touches Postgres; stale cache is acceptable |
| Cache invalidation on rollback | `clear_active_deployment()` deletes both Redis keys atomically |
| `ProjectNotFound` → 404 | Unknown project names never reach the index |
| Watchdog skips healthy deployments | Watchdog does not roll back deployments that pass thresholds |
| Watchdog rolls back on recall breach | A low-recall eval triggers ROLLED_BACK automatically |

**Adding a new failure mode test:**
1. Add a test class in `tests/e2e/test_failure_modes.py` (or a new file).
2. Use `e2e_session` fixture for a real DB session, `load_project` for a pre-created project.
3. Assert the final DB state or error response — do not assert timing (use load tests for that).

### Load Tests (`make test-load`)

```bash
uv run pytest tests/load/ -v --tb=short
```

- **Requires:** Postgres + Redis + Qdrant (`make infra && make migrate`).
- **Run time:** 30–120 s depending on test.
- **What each file proves:**

| File | Claim it proves |
|---|---|
| `test_query_latency.py` | Qdrant ANN p99 < 50 ms on 10 k vectors; cache hit p99 < 20 ms; cache miss p95 < 100 ms |
| `test_concurrent_load.py` | ≥ 50 QPS sustained on cache-miss path; ≥ 500 QPS on cache-hit path |
| `test_ingestion_throughput.py` | ≥ 1 000 vectors/sec upsert to Qdrant |
| `test_deployment_lifecycle.py` | Zero HTTP errors during live config switch; rollback clears Redis key < 2 s |
| `test_cache_efficiency.py` | Warm-cache Qdrant calls == 0; warm QPS ≥ 5 × cold QPS |
| `test_ingestion_dedup.py` | Second ingest job for same config version skips embedding entirely |
| `test_sla_timeout.py` | Hanging backend → HTTP 504 within `QUERY_TIMEOUT_SECONDS + 1 s`; fast query unaffected |
| `test_eval_rollback_workflow.py` | Auto-eval is queued on activation; low-recall eval triggers ROLLED_BACK under query load |

**Important notes:**
- **Embedding latency is excluded** from QPS measurements. The load tests patch `embed_text` with a stub vector. Real embedding adds ~5–50 ms depending on model size and hardware — treat that as additive to load-test numbers.
- **Concurrency saturation point:** A single Qdrant node saturates at roughly 10 concurrent ANN queries on the default resource profile. Beyond that, p99 latency rises sharply. Scale horizontally (multiple Qdrant nodes + shard) before hitting production traffic.
- **Interpreting summary output:** `pytest -v` prints each test's wall-clock duration. Tests that assert specific QPS targets print their measured throughput in `pytest.fail()` messages when thresholds are not met — check those for the actual numbers.

---

## Writing Tests

### Unit tests (no DB)

```python
# tests/unit/test_my_feature.py
import pytest

def test_simple_case() -> None:
    assert my_function("input") == "expected"

class TestMyClass:
    def test_valid_input(self) -> None:
        result = MyClass(foo="bar")
        assert result.foo == "bar"
```

### Mocking async dependencies

The main pain points are `get_redis()` and `get_db()` — both are async functions. Mock them with `AsyncMock`:

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_cache_miss() -> None:
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    # Note: patch with AsyncMock(return_value=...) to make the function itself awaitable
    with patch("retrieval_os.serving.cache.get_redis", AsyncMock(return_value=mock_redis)):
        result = await cache_get("my-plan", 1, "query", 10)

    assert result is None
```

### SQLAlchemy ORM objects in tests

Always use the normal constructor — `__new__` bypasses instrumentation:

```python
# WRONG — AttributeError: NoneType has no attribute 'set'
d = Deployment.__new__(Deployment)
d.status = "ACTIVE"

# CORRECT
d = Deployment(
    plan_name="docs",
    index_config_id=uuid.uuid4(),
    index_config_version=1,
    status="ACTIVE",
    created_at=datetime.now(UTC),
    updated_at=datetime.now(UTC),
    created_by="test",
)
```

---

## Linting and Formatting

We use `ruff` for both linting and formatting:

```bash
# Check
uv run ruff check src tests

# Auto-fix
uv run ruff check src tests --fix

# Format
uv run ruff format src tests
```

Ruff enforces:
- `E/W` — pycodestyle errors and warnings
- `F` — pyflakes (unused imports, undefined names)
- `I` — isort (import ordering)
- `UP` — pyupgrade (use modern Python idioms: `datetime.UTC`, `StrEnum`, etc.)
- `B` — flake8-bugbear (common bugs)

Violations will fail CI. Run `make fmt` before pushing.

---

## Adding a New Domain

Each domain follows the same four-layer pattern. Here is the checklist for adding `lineage/`:

1. **Create the package:**
   ```bash
   mkdir src/retrieval_os/lineage
   touch src/retrieval_os/lineage/__init__.py
   ```

2. **Write the ORM models** (`models.py`)
   - Inherit from `Base` (`retrieval_os.core.database`)
   - Use `Mapped[]` annotations (SQLAlchemy 2.0 style)
   - UUIDv7 default for all PKs
   - Use `StrEnum` for any enum columns

3. **Write a migration** (`alembic/versions/000N_name.py`)
   - Follow the numbering: `0005_lineage.py`
   - Run `make migrate-new NAME=lineage` to generate the file, then fill it in

4. **Write Pydantic schemas** (`schemas.py`)
   - Request schemas: `Create*Request`, `Update*Request`
   - Response schemas: `*Response(ConfigDict(from_attributes=True))`
   - Use `Field(..., description="...")` on all public fields

5. **Write the repository** (`repository.py`)
   - Pure DB access; no business logic
   - All methods take `AsyncSession` as first argument
   - Module-level singleton: `repo = MyRepository()`

6. **Write the service** (`service.py`)
   - Orchestrates repository + any cross-domain calls
   - Returns Pydantic response models (never ORM objects)
   - Raises typed exceptions from `retrieval_os.core.exceptions`
   - Updates Prometheus metrics after successful mutations

7. **Write the router** (`router.py`)
   - `APIRouter(prefix="/v1/...", tags=["..."])`
   - All handlers are thin: parse → call service → return

8. **Wire the router** (`src/retrieval_os/api/main.py`)
   - Import and `app.include_router(...)` in `create_app()`

9. **Add metrics** (`src/retrieval_os/core/metrics.py`)
   - Follow existing naming: `retrieval_os_{domain}_{metric}_{unit}`
   - Add to the relevant section with a comment

10. **Write unit tests** (`tests/unit/test_{domain}.py`)
    - Cover model properties, schema validation, any pure functions
    - Integration tests (with live DB) go in `tests/integration/`

---

## Database Migrations

Migrations use Alembic with async SQLAlchemy.

```bash
# Apply all pending migrations
make migrate

# Create a new migration (autogenerate from models)
make migrate-new NAME=add_eval_jobs

# Check current revision
uv run alembic current

# History
uv run alembic history

# Downgrade one revision
uv run alembic downgrade -1
```

Migration naming convention: `000N_description.py` where N is the next integer.

The `down_revision` field must point to the previous migration's revision ID.

**Important:** Never use `autogenerate` for ARRAY, JSONB, or custom type columns — write them by hand to avoid Alembic's incorrect diffs.

---

## Working with Redis Locally

```bash
# Connect to the local Redis
docker exec -it $(docker ps -q -f name=redis) redis-cli

# See all plan config keys
KEYS ros:plan:*

# Inspect a plan config
GET ros:plan:my-docs:current

# Clear a cache entry
DEL ros:qcache:<sha256>

# Clear all query cache
KEYS ros:qcache:* | xargs redis-cli DEL
```

---

## Working with Qdrant Locally

The Qdrant dashboard is available at: `http://localhost:6333/dashboard`

```bash
# List collections
curl http://localhost:6333/collections

# Create a collection (needed before deploying a plan against it)
curl -X PUT http://localhost:6333/collections/my_docs_v1 \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 1024,
      "distance": "Cosine"
    }
  }'

# Upsert a point
curl -X PUT http://localhost:6333/collections/my_docs_v1/points \
  -H 'Content-Type: application/json' \
  -d '{
    "points": [{
      "id": "abc123",
      "vector": [0.1, 0.2, ...],
      "payload": {
        "text": "This is a sample document chunk",
        "source": "s3://my-bucket/doc.pdf"
      }
    }]
  }'
```

---

## Environment Configuration

See [configuration.md](./configuration.md) for the full reference. The minimum set to override for non-local deployments:

```env
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/retrieval_os
REDIS_URL=redis://user:pass@host:6379/0
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=<key>
S3_SECRET_ACCESS_KEY=<secret>
QDRANT_HOST=qdrant.internal
QDRANT_API_KEY=<key>
OTEL_ENDPOINT=http://otel-collector:4317
```

---

## Observability During Development

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |
| Jaeger | http://localhost:16686 | None |
| Qdrant Dashboard | http://localhost:6333/dashboard | None |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| API Swagger docs | http://localhost:8000/docs | Only in `debug=true` mode |

Query the API and watch traces appear in Jaeger within ~1s.

See [observability.md](./observability.md) for the complete Prometheus metric catalogue and Grafana dashboard descriptions.
