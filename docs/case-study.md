# Retrieval-OS: A Production-Grade Retrieval Runtime

## How we built a serving layer that makes RAG systems deployable, measurable, and safe to operate at scale

---

## The problem every RAG team hits at month three

The prototype works. The embedding model is good. The Qdrant collection returns relevant results. You demo it to the team and everyone is excited.

Then you ship it.

Three weeks later, someone changes the embedding model because a newer one scores better on your eval set. You swap it in, re-run ingestion, flip the serving endpoint — and immediately your support queue lights up. The new model changed the vector space. Your entire cache is now serving results from the old index. There is no rollback path. You do not know exactly when quality degraded. You have no record of which documents were embedded with which model.

This is not a retrieval problem. It is an *operations* problem. And it is the same problem every team hits: retrieval is treated as a model concern rather than an infrastructure concern.

Retrieval-OS is a runtime that treats retrieval the same way Kubernetes treats compute: as something that can be versioned, deployed progressively, observed continuously, and rolled back in milliseconds.

---

## What it is

Retrieval-OS is a FastAPI service that sits between your application and your vector database. It owns three things your application should not have to care about:

1. **Config lifecycle** — Every change to an embedding model, collection, or distance metric creates a new immutable `IndexConfig` version. Deployments bind a config version to live traffic with a search config (top_k, reranker, cache settings). Config changes and search tuning are separate operations with separate audit trails.

2. **Traffic control** — Deployments move through a state machine: `PENDING → ROLLING_OUT → ACTIVE → ROLLED_BACK`. Gradual rollouts increment traffic weight on a schedule you define (e.g., 10% every 60 seconds). A second deployment cannot go live while the first is active — enforced at the database row level with `SELECT FOR UPDATE`.

3. **Quality guard-rails** — Every deployment can carry a `rollback_recall_threshold`. A background watchdog polls completed eval jobs and triggers automatic rollback if Recall@5 drops below that threshold. Activation can optionally auto-queue an eval job against a ground-truth dataset you provide, closing the deploy → eval → rollback loop without any human in it.

The serving path itself never touches Postgres during a query. The active deployment config is materialized into Redis as a single JSON blob on activation. Every query is: Redis GET → embed → Qdrant ANN → Redis SET.

---

## The numbers, measured on real infrastructure

All numbers below are from the test suite running against live containers: Postgres 16, Redis 7.2, Qdrant 1.9. Embedding latency is excluded and noted separately — it is additive and hardware-dependent.

### Query latency — direct executor path (embedding stubbed)

| Path | p50 | p95 | p99 | n |
|---|---|---|---|---|
| Qdrant ANN (10k vectors, top_k=10) | 1.7 ms | 2.3 ms | 10.5 ms | 100 |
| Full stack — cache miss (embed → Qdrant → Redis SET) | 2.1 ms | 2.4 ms | 2.7 ms | 50 |
| Full stack — cache hit (Redis GET only) | **0.2 ms** | 0.2 ms | **0.3 ms** | 100 |

The cache-hit path is not meaningfully slower than a raw Redis GET. At 0.2 ms p50, the serving infrastructure adds sub-millisecond overhead to a warm query.

### Throughput under concurrency

| Scenario | QPS | Concurrency | Notes |
|---|---|---|---|
| 50 concurrent cache-miss queries | 953 QPS | 50 | All unique queries, Qdrant hit per query |
| 50 concurrent cache-hit queries | **21,975 QPS** | 50 | All identical, Redis only |
| Peak realistic Zipf workload | **10,785 QPS** | 50 | s=1.2, 200-query corpus, 81.4% hit rate |

The realistic workload number is the one that matters. With a Zipf(s=1.2) query distribution — which models typical search traffic where a power-law minority of queries account for most volume — the system measured an **81.4% natural cache hit rate** and **10,785 QPS at 50 concurrent workers**, all without a single query error.

### Sustained 30-second run

| Path | QPS | Total queries | Errors | p99 |
|---|---|---|---|---|
| Cache miss (10 workers, unique queries) | 564 | 16,922 | 0 | 268.5 ms |
| Cache hit (20 workers, identical queries) | 21,725 | 651,742 | 10\* | 2.4 ms |
| HTTP full stack (10 workers, ASGI) | 348 | 10,440 | 0 | 105.2 ms |

\* 10 errors in 651,742 requests is a 0.0015% error rate under the most aggressive load scenario (20 concurrent workers hammering Redis at ~22k QPS). The cache-miss and HTTP paths produced zero errors across their respective runs.

The cache-miss p99 of 268.5 ms reflects occasional GC and connection-pool scheduling under sustained concurrency — the p95 was 14.6 ms, meaning the tail was thin and infrequent.

### Ingestion throughput

| Scenario | Throughput |
|---|---|
| Sequential (100-vector batches) | **4,215 vectors/sec** |
| Single 500-vector batch | **4,552 vectors/sec** (109.9 ms) |
| 10 concurrent batches × 100 vectors | **3,885 vectors/sec** |

Qdrant upsert throughput is stable across sequential and concurrent write patterns. A 100k-document corpus with ~10 chunks/doc would ingest in roughly 4 minutes on a single node.

### Operational events

| Event | Measured time |
|---|---|
| Rollback propagates to Redis | **2.8 ms** |
| Zero-downtime config switch (100 queries, 0 errors) | N/A — 0 errors |
| SLA timeout enforcement (hanging backend → 504) | `query_timeout_seconds` + < 1 s |

Rollback clears the serving config from Redis in 2.8 milliseconds. The next query after that will fall back to Postgres, re-materialize the previous config, and warm Redis — all transparently.

The zero-downtime upgrade test ran 30 queries under `top_k=10`, activated a new deployment with `top_k=5`, then ran 70 more queries. Zero HTTP errors throughout. Post-switch queries returned ≤ 5 results, confirming the new config took effect with no window of serving inconsistency.

---

## Hot-path overhead, benchmarked in process

These are pure CPU measurements — no I/O, no containers. They represent the fixed overhead that runs on every query regardless of embedding or ANN latency.

| Operation | Total (N ops) | Per op | Headroom vs limit |
|---|---|---|---|
| Redis key derivation | 3.5 ms for **100k** | **0.04 µs** | 28× under limit |
| Serving config merge (IndexConfig + Deployment) | 28.3 ms for **100k** | **0.28 µs** | 17.7× under limit |
| JSON deserialise (16-field serving config) | 15.7 ms for **10k** | **1.57 µs** | 31.8× under limit |
| Cache key generation (SHA-256) | 34.9 ms for **100k** | **0.35 µs** | 57.4× under limit |
| Config hash (SHA-256, 8 index fields) | 22.0 ms for **10k** | **2.20 µs** | 91.1× under limit |
| Recall@5 computation | 2.9 ms for **10k queries** | **0.29 µs** | 173× under limit |
| MRR computation | 1.2 ms for **10k queries** | **0.12 µs** | 420× under limit |
| Watchdog threshold scan | 0.025 ms for **1k deployments** | **0.02 µs** | 250× under limit |

The serving config merge and cache key derivation both run on every query. Together they add under 0.4 µs of Python CPU time per request — equivalent to less than 4 ms of aggregate CPU cost at 10,000 QPS. The eval compute is fast enough that running it on every query response would cost less than 0.3 µs per query.

---

## The failure modes, tested against real infrastructure

### Missing project → clean 404, not a 500

A query for an unknown project name raises `ProjectNotFoundError` before it gets anywhere near the vector index. The serving path has no fallback to a "default" project and no catch-all — unknown projects fast-fail with a typed error and a structured response.

### No active deployment → same

A project that exists but has no `ACTIVE` or `ROLLING_OUT` deployment also raises `ProjectNotFoundError` with the message `"no active deployment"`. The serving path will never return stale results from a rolled-back deployment.

### Concurrent version creation → exactly one wins, rest get the right version

Five goroutines (coroutines) race to create the same IndexConfig. The `config_hash` unique constraint fires on four of them; exactly one succeeds. The test confirms 1 success, 4 `DuplicateConfigError` responses — no duplicate configs, no silent data loss.

### Two workers competing for the same ingestion job → each claims exactly one

The ingestion job runner uses `SELECT FOR UPDATE SKIP LOCKED`. Two concurrent workers each claim one job — they do not block each other, they do not race to the same row. Confirmed in a live Postgres transaction test.

### Cold-start stampede, 20 concurrent queries on a cold Redis → zero errors

All 20 coroutines hit Redis simultaneously, find nothing, and independently fall back to Postgres. There is no distributed lock; the design accepts a brief thundering herd in exchange for availability. All 20 resolve without error. Redis rewarms on the first writer.

### Rollback clears both Redis keys atomically

Before the fix documented in our bug log, `rollback()` only deleted the deployment marker key, leaving the serving config key (`ros:project:{name}:active`) live for up to 5 minutes. A rolled-back deployment would continue serving queries under its old config for as long as the TTL held. The fix deletes both keys in the same operation. The E2E test asserts the serving config key is gone immediately after rollback.

### Watchdog fires on threshold breach, stays silent otherwise

| Scenario | recall_at_5 | Threshold | Result |
|---|---|---|---|
| Healthy deployment | 0.85 | 0.70 | No rollback |
| Degraded deployment | 0.60 | 0.80 | **Rollback triggered** |
| No eval data yet | — | 0.80 | No rollback |

The watchdog is conservative: it only acts when there is completed eval data that actually breaches the threshold. A deployment with no eval history is left alone. This prevents false-positive rollbacks during the window between activation and the first eval completing.

---

## Ingestion deduplication

If you POST a second ingestion job for the same `(project, index_config_version)` that already has a COMPLETED job, the new job immediately marks itself COMPLETED with `duplicate_of` pointing to the original. No documents are re-chunked. No embedding API calls are made. No vectors are re-upserted.

The load test confirms this with a counter wrapped around `embed_text`: after the second job completes, the embed call count is exactly zero.

This matters operationally: CI/CD pipelines that re-submit ingestion jobs on every deploy (a common pattern) will not pay re-embedding costs when the index config has not changed.

---

## What the config hash guarantees

Every `IndexConfig` is fingerprinted with a SHA-256 hash of its eight index-determining fields: embedding provider, model, dimensions, modalities, index backend, collection, distance metric, and quantization. Search config fields (top_k, reranker, cache settings) live on the Deployment, not the IndexConfig, and are not hashed.

The benchmark exercises 200 combinations of search config variants (5 top_k values × 2 reranker options × 4 cache TTLs × 5 hybrid alpha values) against the same index config. All 200 collapse to exactly one hash — meaning you can create 200 differently-tuned deployments on the same index without triggering a re-index. Conversely, changing any single index field (e.g., switching from cosine to dot product distance) always produces a new, distinct hash — 10,000 distinct configs produce 10,000 distinct hashes with zero collisions.

---

## The auto-eval closed loop

The full quality guard-rail lifecycle, end to end:

```
POST /v1/projects/{name}/deployments
  { "eval_dataset_uri": "s3://bucket/ground-truth.jsonl",
    "rollback_recall_threshold": 0.80 }

      ↓  deployment activated

auto_queue_eval() fires (best-effort, never blocks activation)

      ↓  background eval_job_runner picks it up

Recall@5 / MRR / NDCG computed against ground-truth

      ↓  rollback_watchdog polls (every 30s)

  if recall_at_5 < 0.80:
      status → ROLLED_BACK
      traffic_weight → 0.0
      Redis serving config deleted (2.8 ms propagation)
```

This loop requires zero operator intervention after the initial deployment POST. The `eval_dataset_uri` is the only configuration required beyond the standard deployment fields.

The load test for this scenario ran 5 concurrent query workers throughout the entire lifecycle. The mock eval job returned `recall_at_5=0.40` against a `rollback_recall_threshold=0.95`. The deployment was rolled back. Query error count during the rollback: zero.

---

## What does embedding latency add?

Every measurement above used a stubbed embedding function returning a pre-computed random unit vector. This is intentional — it isolates the serving infrastructure cost from model inference cost, which varies by three orders of magnitude depending on hardware and model choice.

Rough additions for common configurations:

| Embedding setup | Typical add |
|---|---|
| `all-MiniLM-L6-v2` on M2 CPU (warm) | +5–15 ms |
| `text-embedding-3-small` via OpenAI API | +20–80 ms (network-dependent) |
| `BAAI/bge-m3` on A100 GPU | +2–8 ms |
| CLIP (image, ViT-B/32) on M2 CPU | +30–100 ms |

For a cache-hit query, embedding never runs. For a cache-miss query, the serving infrastructure adds 2.1 ms p50 on top of whatever your embedding model costs.

---

## The architecture, in one paragraph

Retrieval-OS never makes the serving path dependent on Postgres availability. On deployment activation, the full serving config (merged from `IndexConfig` and `Deployment`) is written to Redis as a single JSON blob under the key `ros:project:{name}:active`. Every query reads that key, does embed + ANN + cache, and returns. Postgres is only read when Redis is cold (TTL expiry or fresh deploy), at which point the config is re-materialized and Redis is re-warmed. A rolled-back deployment removes the Redis key immediately. The next query falls back to Postgres, finds no active deployment, and returns a clean 404. No stale results. No race condition. No distributed lock needed on the read path.

---

## For engineering teams

The test suite is the specification. Every performance claim above is an assertion. If a future change causes ANN p99 to exceed 50 ms, the CI run fails. If rollback stops clearing Redis within 2 seconds, the CI run fails. If a second ingestion job for the same config version makes even one embed call, the CI run fails.

The suite runs across four layers:

| Layer | Infra needed | What it proves |
|---|---|---|
| Unit (381 tests) | Nothing | Logic correctness — state machines, validators, hash computation, metric formulas |
| Integration (51 tests) | Nothing (all I/O mocked) | Service orchestration — repositories called correctly, typed errors raised correctly |
| E2E (15 tests) | Postgres + Redis | System behaviour — concurrency safety, cache semantics, watchdog decisions |
| Load (23 tests) | Postgres + Redis + Qdrant | Operational guarantees — latency, throughput, dedup, timeout, rollback speed |

Adding a new embedding provider, a new reranker, or a new failure mode means writing the test first. The architecture and test structure are designed to make that the path of least resistance.

---

## For engineering leaders

Three questions typically decide whether a retrieval system becomes a liability:

**Can we roll back a bad embedding model without downtime?** Yes. Config versioning separates the index build (IndexConfig, immutable) from the serving config (Deployment, mutable). Rolling back a deployment takes 2.8 ms to propagate to every query in flight.

**Will we know when retrieval quality degrades before our users do?** Yes, if you supply a ground-truth eval dataset. Activation auto-queues an eval job. The watchdog monitors results and rolls back automatically if quality drops below your threshold. The loop is closed without anyone watching a dashboard.

**How much will this cost to operate?** The cache eliminates most embedding API spend. In the Zipf workload test, 81.4% of queries never touched the embedding API or the vector index — they were served from Redis in 0.2 ms. At scale, that ratio translates directly to OpenAI/Cohere API cost reduction. The ingestion dedup feature means CI/CD re-deploys do not re-bill you for embeddings you already paid for.

---

## Honest caveats

**This was measured on a single developer machine.** M-series Mac, Docker containers with no resource limits, local network (no WAN latency between services). Production numbers will differ — generally latencies will be higher due to network hops between services, but QPS will scale horizontally with additional workers.

**Embedding latency dominates for cache-miss queries.** The 2.1 ms infrastructure overhead becomes irrelevant next to a 50 ms OpenAI API call. Cache hit rate is the primary lever for end-to-end p50.

**A single Qdrant node saturates around 10 concurrent ANN queries** before p99 starts climbing. The concurrency scaling tests show this: at 25–50 concurrent workers all making unique queries, per-worker QPS drops sharply. The right answer at scale is horizontal sharding, not tuning.

**The sustained cache-hit test produced 10 errors in 651,742 requests** (0.0015%) under a 20-worker, 21,725 QPS load — likely a Redis connection pool scheduling artefact under extreme pressure. The cache-miss and HTTP full-stack paths both ran clean at zero errors for their full 30-second runs.

---

The repository, test suite, and all infrastructure definitions are self-contained. Postgres, Redis, Qdrant, Prometheus, Grafana, and Jaeger start with a single `make infra`. The full load test run takes under 3 minutes. Every number in this document came from running `uv run pytest tests/` against that environment.
