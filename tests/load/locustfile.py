"""Locust load test for retrieval-os.

Usage::

    # Against a local dev server
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Headless, 50 users, 5-user ramp-up, run for 60 s
    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
        --headless -u 50 -r 5 --run-time 60s

    # Against staging
    locust -f tests/load/locustfile.py --host https://staging.retrieval-os.example.com
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# ── Synthetic query corpus ────────────────────────────────────────────────────

_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does cosine similarity work for embeddings?",
    "Explain the transformer attention mechanism",
    "What are the differences between BM25 and dense retrieval?",
    "How to handle out-of-vocabulary tokens in NLP?",
    "What is quantization in neural networks?",
    "Explain sliding window attention",
    "How does HNSW indexing work?",
    "What is reciprocal rank fusion?",
    "How to evaluate retrieval quality with NDCG?",
    "Semantic search vs keyword search trade-offs",
    "How to tune embedding batch size for throughput?",
    "What is hybrid search in vector databases?",
    "How does Qdrant handle filtered vector search?",
    "Explain circuit breakers in distributed systems",
]

_PLAN_NAMES = ["wiki-search", "docs-search", "code-search"]


class RetrievalUser(HttpUser):
    """Simulates a client performing retrieval queries."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def query_plan(self) -> None:
        """POST /v1/serve/{plan_name}/query — the main hot path."""
        plan = random.choice(_PLAN_NAMES)
        query = random.choice(_QUERIES)
        payload = {"query": query}
        with self.client.post(
            f"/v1/serve/{plan}/query",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 404:
                # Plan not created in this env — mark as success to avoid skewing errors
                resp.success()
            elif resp.status_code >= 500:
                resp.failure(f"Server error: {resp.status_code}")

    @task(2)
    def health_live(self) -> None:
        """GET /health/live — liveness probe."""
        self.client.get("/health/live")

    @task(1)
    def health_ready(self) -> None:
        """GET /health/ready — readiness probe."""
        self.client.get("/health/ready")

    @task(1)
    def list_plans(self) -> None:
        """GET /v1/plans — list all plans."""
        self.client.get("/v1/plans")

    @task(1)
    def cost_summary(self) -> None:
        """GET /v1/intelligence/cost/summary — cost aggregation endpoint."""
        self.client.get("/v1/intelligence/cost/summary")


class AdminUser(HttpUser):
    """Simulates an operator checking recommendations and pricing."""

    wait_time = between(1.0, 3.0)
    weight = 1  # 1 admin per 10 retrieval users (adjust with --users ratio)

    @task(3)
    def get_recommendations(self) -> None:
        self.client.get("/v1/intelligence/recommendations")

    @task(2)
    def list_model_pricing(self) -> None:
        self.client.get("/v1/intelligence/model-pricing")

    @task(1)
    def list_deployments(self) -> None:
        self.client.get("/v1/deployments")

    @task(1)
    def list_eval_jobs(self) -> None:
        plan = random.choice(_PLAN_NAMES)
        self.client.get(f"/v1/evals?plan_name={plan}")
