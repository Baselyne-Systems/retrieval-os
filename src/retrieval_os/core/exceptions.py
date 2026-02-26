"""Typed exception hierarchy for retrieval-os.

All exceptions carry an HTTP status code and a machine-readable error_code so
the global exception handler can produce consistent JSON error responses.
"""


class RetrievalOSError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


# ── 404 Not Found ─────────────────────────────────────────────────────────────


class NotFoundError(RetrievalOSError):
    status_code = 404
    error_code = "NOT_FOUND"


class ProjectNotFoundError(NotFoundError):
    error_code = "PROJECT_NOT_FOUND"


class IndexConfigNotFoundError(NotFoundError):
    error_code = "INDEX_CONFIG_NOT_FOUND"


class DeploymentNotFoundError(NotFoundError):
    error_code = "DEPLOYMENT_NOT_FOUND"


class ArtifactNotFoundError(NotFoundError):
    error_code = "ARTIFACT_NOT_FOUND"


class ArtifactStorageNotFoundError(NotFoundError):
    error_code = "ARTIFACT_STORAGE_NOT_FOUND"


class EvalJobNotFoundError(NotFoundError):
    error_code = "EVAL_JOB_NOT_FOUND"


# ── 409 Conflict ──────────────────────────────────────────────────────────────


class ConflictError(RetrievalOSError):
    status_code = 409
    error_code = "CONFLICT"


class DuplicateConfigError(ConflictError):
    error_code = "DUPLICATE_CONFIG_HASH"


class DeploymentStateError(ConflictError):
    error_code = "DEPLOYMENT_STATE_ERROR"


class DeploymentLockError(ConflictError):
    error_code = "DEPLOYMENT_LOCK_CONFLICT"


# ── 422 Validation ────────────────────────────────────────────────────────────


class AppValidationError(RetrievalOSError):
    status_code = 422
    error_code = "VALIDATION_ERROR"


class LineageCycleError(AppValidationError):
    error_code = "LINEAGE_CYCLE_DETECTED"


# ── 503 Upstream Unavailable ──────────────────────────────────────────────────


class UpstreamError(RetrievalOSError):
    status_code = 503
    error_code = "UPSTREAM_ERROR"


class IndexBackendError(UpstreamError):
    error_code = "INDEX_BACKEND_ERROR"


class EmbeddingProviderError(UpstreamError):
    error_code = "EMBEDDING_PROVIDER_ERROR"


class CircuitOpenError(UpstreamError):
    error_code = "CIRCUIT_BREAKER_OPEN"


# ── 401 Authentication ─────────────────────────────────────────────────────────


class AuthenticationError(RetrievalOSError):
    status_code = 401
    error_code = "AUTHENTICATION_REQUIRED"


# ── 429 Rate Limit ────────────────────────────────────────────────────────────


class RateLimitError(RetrievalOSError):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
