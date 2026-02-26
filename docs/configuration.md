# Configuration Reference

All configuration is provided via environment variables, a `.env` file in the project root, or both. Variables in the environment take precedence over `.env`. Names are case-insensitive.

Copy `.env.example` to `.env` to start:

```bash
cp .env.example .env
```

---

## Application

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_NAME` | string | `retrieval-os` | Service name, used in OTel resource attributes and log context. |
| `APP_VERSION` | string | `0.1.0` | Semver version string. Returned by `GET /v1/info`. |
| `ENVIRONMENT` | string | `development` | Environment name. Use `production`, `staging`, `development`. Affects: `debug` mode, Swagger UI visibility. |
| `DEBUG` | bool | `false` | Enables Swagger UI (`/docs`, `/redoc`) and verbose SQLAlchemy query logging. **Never enable in production.** |
| `LOG_LEVEL` | string | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |

---

## Database (PostgreSQL)

| Variable | Type | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | string | `postgresql+asyncpg://retrieval_os:retrieval_os@localhost:5432/retrieval_os` | SQLAlchemy async DSN. Must use the `+asyncpg` driver. |
| `DATABASE_POOL_SIZE` | int | `10` | Number of persistent connections in the pool. Set to `(num_workers * 2)` as a starting point. |
| `DATABASE_MAX_OVERFLOW` | int | `20` | Additional connections allowed beyond `POOL_SIZE` during traffic bursts. Total max = `POOL_SIZE + MAX_OVERFLOW`. |
| `DATABASE_POOL_TIMEOUT` | int | `30` | Seconds to wait for a connection before raising `TimeoutError`. |
| `DATABASE_ECHO` | bool | `false` | Log all SQL statements. Implies DEBUG verbosity. |

**Connection string format:**

```
postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}
```

For managed Postgres (e.g. RDS, Cloud SQL) with SSL:

```
postgresql+asyncpg://user:pass@host:5432/db?ssl=require
```

---

## Redis

| Variable | Type | Default | Description |
|---|---|---|---|
| `REDIS_URL` | string | `redis://localhost:6379/0` | Redis connection URL. Use `redis://` for plain, `rediss://` for TLS. |
| `REDIS_POOL_SIZE` | int | `10` | Maximum connections in the async pool. |
| `REDIS_SOCKET_TIMEOUT` | float | `5.0` | Socket timeout in seconds. Commands that exceed this raise `TimeoutError`. |

**Redis URL formats:**

```
redis://localhost:6379/0                    # plain, db index 0
redis://user:password@host:6379/0          # with auth
rediss://user:password@host:6380/0         # TLS
redis://localhost:6379/1                   # different db index
```

Redis is used for:
- Semantic query cache (`ros:qcache:*`) — TTL per plan
- Plan config warm cache (`ros:plan:*:current`) — 30s TTL, refreshed on cache miss
- Active deployment marker (`ros:deployment:*:active`) — 5min TTL

---

## S3 / MinIO

| Variable | Type | Default | Description |
|---|---|---|---|
| `S3_ENDPOINT_URL` | string | `http://localhost:9000` | S3-compatible endpoint. Set to `https://s3.amazonaws.com` for AWS S3. |
| `S3_ACCESS_KEY_ID` | string | `minioadmin` | Access key ID. For AWS, use an IAM access key. |
| `S3_SECRET_ACCESS_KEY` | string | `minioadmin` | Secret access key. |
| `S3_BUCKET_NAME` | string | `retrieval-os` | Bucket for artifact storage. Created on startup if it doesn't exist (MinIO only; AWS requires pre-creation). |
| `S3_REGION` | string | `us-east-1` | AWS region. Ignored by MinIO. |

**AWS setup:**

1. Create a bucket: `aws s3 mb s3://your-bucket --region us-east-1`
2. Create an IAM user with `s3:GetObject`, `s3:PutObject`, `s3:HeadObject` on `arn:aws:s3:::your-bucket/*`
3. Set `S3_ENDPOINT_URL=https://s3.amazonaws.com` (or remove it; boto3 defaults to AWS)

**GCS setup:**

Use the GCS S3-compatibility endpoint:

```
S3_ENDPOINT_URL=https://storage.googleapis.com
S3_ACCESS_KEY_ID=<HMAC key>
S3_SECRET_ACCESS_KEY=<HMAC secret>
```

---

## Qdrant

| Variable | Type | Default | Description |
|---|---|---|---|
| `QDRANT_HOST` | string | `localhost` | Qdrant server hostname. |
| `QDRANT_GRPC_PORT` | int | `6334` | gRPC port (preferred for production — lower overhead). |
| `QDRANT_HTTP_PORT` | int | `6333` | HTTP/REST port (used by Qdrant dashboard and health checks). |
| `QDRANT_API_KEY` | string | `null` | API key for authenticated Qdrant Cloud or self-hosted with auth enabled. |

Retrieval-OS uses the gRPC API (`QDRANT_GRPC_PORT`) for all query traffic. The HTTP port is not used at runtime but is documented for tooling.

For Qdrant Cloud:

```
QDRANT_HOST=abc123.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=your-api-key
```

---

## Observability (OpenTelemetry)

| Variable | Type | Default | Description |
|---|---|---|---|
| `OTEL_ENDPOINT` | string | `http://localhost:4317` | OTLP gRPC endpoint for the OTel collector or Jaeger. |
| `OTEL_ENABLED` | bool | `true` | Disable to suppress all trace export (e.g. in CI/test runs). |
| `OTEL_SERVICE_NAME` | string | `retrieval-os-api` | Service name that appears in Jaeger and other trace backends. |

The local docker-compose stack runs Jaeger all-in-one which accepts OTLP gRPC directly on port 4317:

```
OTEL_ENDPOINT=http://localhost:4317
```

For production, route through an OTel Collector to batch and sample:

```
OTEL_ENDPOINT=http://otel-collector:4317
```

To disable tracing entirely in CI:

```
OTEL_ENABLED=false
```

---

## Auth & Rate Limiting

| Variable | Type | Default | Description |
|---|---|---|---|
| `AUTH_ENABLED` | bool | `false` | Enable API key authentication. When `true`, every request must include a valid key in the header specified by `API_KEY_HEADER`. |
| `API_KEY_HEADER` | string | `X-API-Key` | HTTP header name for the API key. |
| `RATE_LIMIT_ENABLED` | bool | `false` | Enable per-tenant sliding-window rate limiting via Redis. Requires `AUTH_ENABLED=true`. |
| `RATE_LIMIT_DEFAULT_RPM` | int | `60` | Default max requests per minute per tenant. Overridden per-tenant by the `rate_limit_rpm` field on the `Tenant` record. |

**Notes:**

- Auth and rate limiting are disabled by default for local development. Always enable `AUTH_ENABLED=true` in staging and production.
- The API key is hashed with SHA-256 on write; only the hash is stored. There is no way to retrieve a raw key after creation.
- Rate limit state is stored in Redis as a sorted set (`ros:ratelimit:{tenant_id}`). Entries expire automatically after 1 minute.

---

## Optional External API Keys

| Variable | Type | Default | Description |
|---|---|---|---|
| `COHERE_API_KEY` | string | `null` | Cohere API key. Required only when using the `cohere` reranker in a plan's `reranker` field (e.g. `cohere:rerank-english-v3.0`). |

---

## Background Task Intervals

These control how often each asyncio background loop runs. All five loops are started in the FastAPI lifespan and run indefinitely until the process shuts down.

| Variable | Type | Default | Description |
|---|---|---|---|
| `ROLLBACK_WATCHDOG_INTERVAL_SECONDS` | int | `30` | How often the rollback watchdog checks active deployments against their guard-rail thresholds. |
| `ROLLOUT_STEPPER_INTERVAL_SECONDS` | int | `10` | How often the rollout stepper advances `ROLLING_OUT` deployments by one step. |
| `EVAL_JOB_POLL_INTERVAL_SECONDS` | int | `5` | How often the eval job runner polls for `QUEUED` eval jobs. |
| `COST_AGGREGATOR_INTERVAL_SECONDS` | int | `3600` | How often usage records are aggregated into hourly cost entries. |
| `INGESTION_JOB_POLL_INTERVAL_SECONDS` | int | `5` | How often the ingestion job runner polls for `QUEUED` ingestion jobs. |

**Tuning notes:**

- `ROLLBACK_WATCHDOG_INTERVAL_SECONDS`: The watchdog compares the latest completed eval's metrics against each deployment's thresholds. Lowering this to 15s in production gives faster auto-rollback response. The watchdog has no cost beyond a few DB reads per cycle.
- `ROLLOUT_STEPPER_INTERVAL_SECONDS` is the granularity of gradual rollouts. If a deployment has `rollout_step_interval_seconds=300` (5 min), the stepper will only advance it when the loop runs AND 5 minutes have elapsed since the last step. Setting this to 60s gives 1-minute precision on 5-minute steps.
- `EVAL_JOB_POLL_INTERVAL_SECONDS` only affects how quickly a newly-queued job starts. 5s is aggressive; 30s is reasonable for lower-volume setups.
- `COST_AGGREGATOR_INTERVAL_SECONDS`: 3600 (hourly) is standard. Real-time cost dashboards read from `usage_records` directly.
- `INGESTION_JOB_POLL_INTERVAL_SECONDS`: At 5s, a new job starts within 5 seconds of submission. For high-throughput pipelines, consider running multiple API replicas — each replica runs its own job runner, and `SELECT FOR UPDATE SKIP LOCKED` ensures no job is processed twice.

---

## Per-Environment Configs

### Local development (`.env`)

```env
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql+asyncpg://retrieval_os:retrieval_os@localhost:5432/retrieval_os
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin
S3_BUCKET_NAME=retrieval-os
QDRANT_HOST=localhost
OTEL_ENDPOINT=http://localhost:4317
OTEL_ENABLED=true
```

### Staging

```env
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql+asyncpg://user:pass@staging-pg.internal:5432/retrieval_os
REDIS_URL=redis://staging-redis.internal:6379/0
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=AKIA...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=retrieval-os-staging
QDRANT_HOST=qdrant.staging.internal
QDRANT_API_KEY=...
OTEL_ENDPOINT=http://otel-collector.staging.internal:4317
AUTH_ENABLED=true
RATE_LIMIT_ENABLED=true
```

### Production

```env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql+asyncpg://user:pass@prod-pg.internal:5432/retrieval_os
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
REDIS_URL=rediss://user:pass@prod-redis.internal:6380/0
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=AKIA...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET_NAME=retrieval-os-prod
QDRANT_HOST=qdrant.prod.internal
QDRANT_API_KEY=...
OTEL_ENDPOINT=http://otel-collector.prod.internal:4317
AUTH_ENABLED=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT_RPM=120
ROLLBACK_WATCHDOG_INTERVAL_SECONDS=15
# COHERE_API_KEY=...  # only needed if any plan uses reranker: "cohere:..."
```

---

## Secrets Management

Do not commit `.env` or any file containing real credentials. The `.gitignore` excludes `.env` by default.

For production, inject secrets as environment variables from your secrets manager:

```bash
# AWS Secrets Manager example
export DATABASE_URL=$(aws secretsmanager get-secret-value \
  --secret-id prod/retrieval-os/database-url \
  --query SecretString --output text)
```

For Kubernetes, use a `Secret` mounted as environment variables:

```yaml
envFrom:
  - secretRef:
      name: retrieval-os-secrets
```
