# Retrieval-OS

A production-grade serving layer for multimodal retrieval-augmented systems. Sits above vector databases and embedding model APIs to make retrieval coherent, observable, safe to change, and measurable.

Think: **CI/CD for retrieval**. Deployment control for embedding and index changes. Continuous quality measurement via Recall@k, MRR, NDCG. Runtime cost and latency intelligence.

---

## What It Solves

Retrieval-augmented systems fail silently. You swap an embedding model, rebuild your index with slightly different settings, or adjust top-k — and recall drops two weeks later with no attribution. There is no equivalent of a diff for retrieval config. No controlled rollout. No automatic regression detection.

Retrieval-OS provides:

- **Immutable versioned configs** (Plans) — every change is tracked, every version is reachable
- **Controlled deployments** — gradual rollouts with traffic weights, automatic rollback on threshold breach
- **Continuous evaluation** — Recall@k, MRR, NDCG measured against ground truth on every deploy
- **Full lineage** — from training dataset to deployed index, queryable in one API call
- **Cost intelligence** — per-plan cost aggregation, cache efficiency, actionable optimization recommendations

---

## Stack

```
PostgreSQL 16   Redis 7.2   Qdrant   MinIO/S3
Prometheus      Grafana     Jaeger   OpenTelemetry
Python 3.12 + FastAPI + asyncio   (no Kafka, no Celery)
```

No message broker. No separate worker process. All background work runs as asyncio tasks in the FastAPI lifespan.

---

## Quick Start

```bash
# Prerequisites: Python 3.12+, uv, Docker
git clone https://github.com/your-org/retrieval-os.git
cd retrieval-os

uv sync --all-extras
cp .env.example .env
make infra        # starts Postgres, Redis, Qdrant, MinIO, Prometheus, Grafana, Jaeger
make migrate      # runs Alembic migrations
make dev          # starts the API on :8000 with hot reload
```

```bash
# Verify the stack
curl http://localhost:8000/ready
# {"status":"ok","checks":{"postgres":"ok","redis":"ok"}}
```

Create a plan, deploy it, and query it:

```bash
# Create a plan
curl -X POST http://localhost:8000/v1/plans \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "docs",
    "created_by": "eng",
    "config": {
      "embedding_provider": "sentence_transformers",
      "embedding_model": "BAAI/bge-m3",
      "embedding_dimensions": 1024,
      "modalities": ["text"],
      "index_backend": "qdrant",
      "index_collection": "docs_v1",
      "distance_metric": "cosine",
      "top_k": 10
    }
  }'

# Deploy it
curl -X POST http://localhost:8000/v1/plans/docs/deployments \
  -H 'Content-Type: application/json' \
  -d '{"plan_version": 1, "created_by": "eng"}'

# Query it
curl -X POST http://localhost:8000/v1/query/docs \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does RAG work?"}'
```

---

## Documentation

| Doc | What it covers |
|---|---|
| [Architecture](docs/architecture.md) | System design, serving path invariants, Redis layout, deployment state machine, config hash, error hierarchy |
| [API Reference](docs/api-reference.md) | All endpoints with request/response schemas, error codes, and curl examples |
| [Data Models](docs/data-models.md) | Schema-level reference for every table, column types, constraints, indexes |
| [Configuration](docs/configuration.md) | All environment variables with defaults, constraints, and per-environment examples |
| [Observability](docs/observability.md) | Complete Prometheus metric catalogue, trace structure, alert rules, Grafana dashboards |
| [Developer Guide](docs/developer-guide.md) | Setup, Make targets, project layout, test patterns, migration workflow |

---

## API Surface (implemented)

```
# Serving (hot path — P99 target < 200ms)
POST  /v1/query/{plan_name}

# Plans (immutable versioned configs)
POST   /v1/plans
GET    /v1/plans                        cursor-paginated
GET    /v1/plans/{name}
POST   /v1/plans/{name}/versions
GET    /v1/plans/{name}/versions
GET    /v1/plans/{name}/versions/{num}
POST   /v1/plans/{name}/clone
DELETE /v1/plans/{name}                 soft-archive

# Deployments (traffic control)
POST   /v1/plans/{name}/deployments
GET    /v1/plans/{name}/deployments
GET    /v1/plans/{name}/deployments/{id}
POST   /v1/plans/{name}/deployments/{id}/rollback

# Infrastructure
GET    /health
GET    /ready
GET    /metrics                         Prometheus text format
GET    /v1/info
```

---

## Build Progress

| Phase | Status | What it adds |
|---|---|---|
| **1 — Foundation** | Done | FastAPI app, infra stack, OTel, Prometheus, structured logs, /health /ready /metrics |
| **2 — Plans** | Done | Immutable versioned plan configs, validation, config hash deduplication |
| **3 — Serving Path** | Done | Redis cache → embed → Qdrant ANN → response; usage record fire-and-forget |
| **4 — Deployments** | Done | Deployment state machine, gradual rollouts, rollback watchdog + stepper activated |
| **5 — Lineage** | Next | Artifact DAG, recursive CTE traversal, orphan detection |
| **6 — Evaluation** | Planned | Recall@k, MRR, NDCG, eval job runner, regression alerts |
| **7 — Cost Intelligence** | Planned | Usage aggregation, cost per plan, optimization recommendations |
| **8 — Production Hardening** | Planned | K8s, circuit breakers, cross-modal RRF, load tests |

**87 unit tests, all passing. Linter clean (ruff).**

---

## Key Design Decisions

**Serving path never reads Postgres.**
Plan config lives in Redis (`ros:plan:{name}:current`, 30s TTL). A Postgres failure cannot cause query failures. On Redis miss, one Postgres read warms the cache.

**Plan validation at write time.**
Every `PlanVersion` in the database is contractually valid. The serving path performs zero defensive checks. Validation errors surface at plan creation with all field failures in one response.

**Config hash for deduplication.**
`compute_config_hash()` produces a SHA-256 of the fields that define retrieval behaviour (excluding cost/cache/governance fields). Creating a version with identical behaviour to an existing one returns HTTP 409 — no wasted DB row, no ambiguous dual versions.

**SELECT FOR UPDATE for monotonic version numbers.**
Concurrent version creates lock the parent `RetrievalPlan` row. This serialises the `MAX(version) + 1` computation and guarantees gapless monotonic integers across racing requests.

**UUIDv7 PKs everywhere.**
Time-ordered, globally unique, embed creation timestamp, no enumeration attack surface.

**Asyncio for all background work.**
Four loops in the FastAPI lifespan: rollback watchdog, rollout stepper, eval job runner, cost aggregator. Each loop commits its own session. Exceptions are logged and swallowed — a transient DB error never crashes the API.

---

## Metrics Available Now

```
retrieval_os_retrieval_requests_total{plan_name}
retrieval_os_retrieval_latency_seconds{plan_name}        P50/P95/P99 SLO
retrieval_os_cache_hits_total{plan_name}
retrieval_os_cache_misses_total{plan_name}
retrieval_os_embed_requests_total{provider}
retrieval_os_embed_latency_seconds{provider}
retrieval_os_embed_errors_total{provider}
retrieval_os_index_latency_seconds{backend}
retrieval_os_index_errors_total{backend}
retrieval_os_plans_total
retrieval_os_plan_versions_total{plan_name}
retrieval_os_deployment_status{deployment_id, plan_name, environment, status}
retrieval_os_deployment_traffic_weight{deployment_id, plan_name, environment}
retrieval_os_rollback_events_total{deployment_id, plan_name, triggered_by}
retrieval_os_rollout_duration_seconds{plan_name}
```

Full trace per query in Jaeger. Structured JSON logs to stdout.
