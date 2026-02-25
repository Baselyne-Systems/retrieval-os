.PHONY: dev infra test lint fmt migrate shell help

PYTHON := python
UV := uv

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  dev       Start full stack (infra + api)"
	@echo "  infra     Start infra only (postgres, redis, qdrant, minio, prometheus, grafana, jaeger)"
	@echo "  stop      Stop all containers"
	@echo "  test      Run test suite"
	@echo "  lint      Run ruff + mypy"
	@echo "  fmt       Auto-format with ruff"
	@echo "  migrate   Run pending Alembic migrations"
	@echo "  shell     Open a Python shell with app context"
	@echo "  install   Install dependencies (requires uv)"

install:
	$(UV) sync --extra dev

dev:
	docker compose up --build

infra:
	docker compose -f docker-compose.infra.yml up -d
	@echo "Waiting for postgres..."
	@until docker compose -f docker-compose.infra.yml exec postgres pg_isready -U retrieval_os; do sleep 1; done
	@echo "Infrastructure ready."

stop:
	docker compose down
	docker compose -f docker-compose.infra.yml down

test:
	$(UV) run pytest --cov=src/retrieval_os --cov-report=term-missing -v

test-unit:
	$(UV) run pytest tests/unit -v

test-integration:
	$(UV) run pytest tests/integration -v

lint:
	$(UV) run ruff check src tests
	$(UV) run mypy src

fmt:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

migrate:
	$(UV) run alembic upgrade head

migrate-new:
	@read -p "Migration name: " name; \
	$(UV) run alembic revision --autogenerate -m "$$name"

shell:
	$(UV) run python -c "from retrieval_os.core.config import settings; import code; code.interact(local=locals())"
