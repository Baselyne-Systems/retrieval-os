"""FastAPI router for the Deployments domain."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.deployments import service
from retrieval_os.deployments.schemas import (
    CreateDeploymentRequest,
    DeploymentResponse,
    RollbackRequest,
)

router = APIRouter(prefix="/v1/projects", tags=["deployments"])


@router.post(
    "/{plan_name}/deployments",
    status_code=201,
    response_model=DeploymentResponse,
)
async def create_deployment(
    plan_name: str,
    request: CreateDeploymentRequest,
    db: AsyncSession = Depends(get_db),
) -> DeploymentResponse:
    """Deploy a specific version of a plan to live traffic.

    Instant deploy: omit rollout_step_percent and rollout_step_interval_seconds.
    Gradual deploy: provide both; the rollout stepper advances traffic each step.
    """
    return await service.create_deployment(db, plan_name, request)


@router.get(
    "/{plan_name}/deployments",
    response_model=list[DeploymentResponse],
)
async def list_deployments(
    plan_name: str,
    db: AsyncSession = Depends(get_db),
) -> list[DeploymentResponse]:
    """List all deployments for a plan, newest first."""
    return await service.list_deployments(db, plan_name)


@router.get(
    "/{plan_name}/deployments/{deployment_id}",
    response_model=DeploymentResponse,
)
async def get_deployment(
    plan_name: str,
    deployment_id: str,
    db: AsyncSession = Depends(get_db),
) -> DeploymentResponse:
    """Get a specific deployment."""
    return await service.get_deployment(db, plan_name, deployment_id)


@router.post(
    "/{plan_name}/deployments/{deployment_id}/rollback",
    response_model=DeploymentResponse,
)
async def rollback_deployment(
    plan_name: str,
    deployment_id: str,
    request: RollbackRequest,
    db: AsyncSession = Depends(get_db),
) -> DeploymentResponse:
    """Immediately roll back a live deployment to 0% traffic."""
    return await service.rollback_deployment(db, plan_name, deployment_id, request)
