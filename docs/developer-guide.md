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
