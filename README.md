# Retrieval-OS

**Production control plane for RAG and semantic search systems.**

Most teams build retrieval into their AI applications and then can't safely change it. A different embedding model, tighter chunking, a new index — these changes go straight to production as a single undifferentiated release. There is no canary. No quality check. No rollback. If something degrades, you find out from users, not from your infrastructure.

Retrieval-OS is the layer that closes that gap. It sits between your application and your vector database, and gives you the same operational controls you have for your application code: versioned configs, staged rollouts, automated quality measurements, and automatic rollback when quality drops.

---

## The Workflow

**1. Define your retrieval pipeline as a Plan.**

A Plan captures everything that determines how your retrieval works: embedding model, chunking settings, index collection, distance metric, top-k, reranking. It is versioned — every change creates a new version, and old versions remain queryable and reproducible.

**2. Ingest your documents.**

Push documents via API. The system chunks them using the Plan's settings, embeds them with the Plan's embedding model, and upserts the vectors into the Plan's index collection. The exact config used is recorded alongside the vectors.

**3. Deploy a version.**

When a new version is ready, deploy it. You choose the mode:
- **Instant** — full traffic switches immediately.
- **Gradual** — start at 10%, increment automatically every N minutes, promote to full traffic when stable.

**4. Measure quality automatically.**

Run evaluation jobs against your labelled queries. The system computes Recall@k, MRR, and NDCG and compares them to the previous deployment. A regression fires a webhook and marks the deployment for review.

**5. Set guard-rails.**

Attach thresholds to a deployment: minimum Recall@5, maximum error rate. The watchdog checks these on every eval cycle. If a threshold is breached, the deployment is rolled back automatically — no pager, no manual intervention.

---

## What You Get

| | |
|---|---|
| **Versioned retrieval configs** | Every change to model, chunking, index, or reranking is a numbered version. Reproduce any historical config exactly. Compare versions. Revert in one API call. |
| **Gradual rollouts** | Traffic-weighted canary deployments that advance automatically. Roll back instantly if something goes wrong. |
| **Automatic quality guard-rails** | Set Recall@5 or error-rate thresholds. Rollback happens automatically when they breach — the eval loop runs continuously without human involvement. |
| **Document ingestion** | REST API for pushing documents. Chunks, embeds, and upserts using the active plan version's exact settings. Lineage is recorded automatically. |
| **Full artifact lineage** | Every document chunk traces back to its source dataset, embedding run, and plan version. Answer "what was in the index when that query was served?" |
| **Retrieval quality metrics** | Recall@k, MRR, NDCG tracked per deployment. Regression detection against the previous baseline. |
| **Cost intelligence** | Per-plan embed token spend, cache hit ratio, and actionable optimisation recommendations. |
| **Multi-tenancy** | API key auth, per-tenant rate limiting, tenant-scoped index isolation. |
| **Observability** | Full OpenTelemetry traces per query, Prometheus metrics, structured JSON logs. |
| **Webhooks** | Signed HMAC-SHA256 event delivery for deployments, rollbacks, eval regressions, and ingestion completions. |

---

## Quick Start

```bash
# Prerequisites: Python 3.12+, uv, Docker
git clone https://github.com/your-org/retrieval-os.git
cd retrieval-os
uv sync --all-extras
cp .env.example .env
make infra        # starts Postgres, Redis, Qdrant, MinIO, Prometheus, Grafana, Jaeger
make migrate
make dev          # API on :8000 with hot reload
```

```bash
# Verify the stack
curl http://localhost:8000/ready
# {"status":"ok","checks":{"postgres":"ok","redis":"ok"}}
```

### Create a plan and run a query

```bash
# 1. Define your retrieval pipeline
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

# 2. Ingest documents
curl -X POST http://localhost:8000/v1/plans/docs/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "created_by": "eng",
    "chunk_size": 512,
    "overlap": 64,
    "documents": [
      {"id": "doc-1", "content": "Your document text here.", "metadata": {"source": "wiki"}}
    ]
  }'

# 3. Deploy version 1
curl -X POST http://localhost:8000/v1/plans/docs/deployments \
  -H 'Content-Type: application/json' \
  -d '{"plan_version": 1, "created_by": "eng"}'

# 4. Query
curl -X POST http://localhost:8000/v1/query/docs \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does RAG work?"}'
```

---

## Documentation

| Doc | What it covers |
|---|---|
| [Architecture](docs/architecture.md) | Concepts and mental model, system design, serving path, deployment state machine, background tasks |
| [API Reference](docs/api-reference.md) | All endpoints with request/response schemas, error codes, and curl examples |
| [Data Models](docs/data-models.md) | Schema-level reference for every table, column types, constraints, indexes |
| [Configuration](docs/configuration.md) | All environment variables with defaults, constraints, and per-environment examples |
| [Observability](docs/observability.md) | Prometheus metric catalogue, trace structure, alert rules, Grafana dashboards |
| [Developer Guide](docs/developer-guide.md) | Setup, Make targets, project layout, test patterns, migration workflow |

---

## Stack

```
PostgreSQL 16   Redis 7.2   Qdrant   MinIO/S3
Prometheus      Grafana     Jaeger   OpenTelemetry
Python 3.12 + FastAPI + asyncio   (no Kafka, no Celery)
```

No message broker. No separate worker process. All background work runs as asyncio tasks in the FastAPI lifespan.

---

## API Surface

```
POST  /v1/query/{plan_name}                            Serve a retrieval query

POST   /v1/plans/{name}/ingest                         Ingest documents
GET    /v1/plans/{name}/ingest/{job_id}                Check ingestion job status

POST   /v1/plans                                       Create a plan
GET    /v1/plans                                       List plans (cursor-paginated)
GET    /v1/plans/{name}                                Get a plan
DELETE /v1/plans/{name}                                Archive a plan

POST   /v1/plans/{name}/versions                       Create a new version
GET    /v1/plans/{name}/versions                       List all versions
GET    /v1/plans/{name}/versions/{num}                 Get a specific version

POST   /v1/plans/{name}/deployments                    Deploy a version
GET    /v1/plans/{name}/deployments                    List deployments
GET    /v1/plans/{name}/deployments/{id}               Get deployment status
POST   /v1/plans/{name}/deployments/{id}/rollback      Roll back a deployment

POST   /v1/webhooks                                    Register a webhook
GET    /v1/webhooks                                    List webhooks
DELETE /v1/webhooks/{id}                               Remove a webhook

GET    /health
GET    /ready
GET    /metrics                                        Prometheus text format
```

---

## Build Status

| Phase | Status | What it adds |
|---|---|---|
| Foundation | Done | FastAPI app, infra stack, OTel, Prometheus, structured logs |
| Plans | Done | Versioned pipeline configs, validation, config hash deduplication |
| Serving | Done | Redis cache → embed → Qdrant ANN → response |
| Deployments | Done | State machine, gradual rollouts, rollback watchdog |
| Lineage | Done | Artifact DAG, dataset → embedding → index traceability |
| Evaluation | Done | Recall@k, MRR, NDCG, eval job runner, regression detection |
| Cost Intelligence | Done | Usage aggregation, cost per plan, optimisation recommendations |
| Production Hardening | Done | Circuit breakers, cross-modal RRF, K8s manifests, load tests |
| Multimodal | Done | CLIP image embed, Whisper audio→text, sparse BM25 |
| Multi-tenancy | Done | API key auth, rate limiting, tenant isolation |
| Webhooks | Done | HMAC-SHA256 signed event delivery with retry |
| Ingestion | Done | Word-boundary chunker, embed→upsert pipeline, lineage auto-registration |

**438 tests (387 unit + 51 integration), all passing. Linter clean (ruff).**
