"""Enumeration of all webhook event types emitted by Retrieval-OS."""

from __future__ import annotations

from enum import StrEnum


class WebhookEvent(StrEnum):
    """Events that can trigger outbound webhook deliveries."""

    DEPLOYMENT_STATUS_CHANGED = "deployment.status_changed"
    EVAL_REGRESSION_DETECTED = "eval.regression_detected"
    PLAN_VERSION_CREATED = "plan.version_created"
    COST_THRESHOLD_EXCEEDED = "cost.threshold_exceeded"
