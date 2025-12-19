"""Health check endpoint for Doctor Cornelius API.

This module provides the health check endpoint that returns the status
of all system components including Neo4j, Slack, and Gemini services.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from doctor_cornelius.config import Settings, get_settings

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


class ComponentStatus(str, Enum):
    """Status values for individual components."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"


class OverallStatus(str, Enum):
    """Overall health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    status: ComponentStatus = Field(description="Component status")
    latency_ms: float | None = Field(default=None, description="Response latency in milliseconds")
    error: str | None = Field(default=None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: OverallStatus = Field(description="Overall system health status")
    components: dict[str, ComponentHealth] = Field(
        description="Health status of individual components"
    )
    version: str = Field(description="Application version")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "components": {
                        "neo4j": {"status": "up", "latency_ms": 12.5},
                        "slack": {"status": "up", "latency_ms": 45.2},
                        "gemini": {"status": "up", "latency_ms": 120.0},
                    },
                    "version": "1.0.0",
                }
            ]
        }
    }


async def _check_neo4j_health(settings: Settings) -> ComponentHealth:
    """Check Neo4j database connectivity.

    Args:
        settings: Application settings containing Neo4j configuration.

    Returns:
        ComponentHealth with the status of the Neo4j connection.
    """
    from neo4j import AsyncGraphDatabase

    start_time = time.perf_counter()
    try:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password.get_secret_value()),
        )
        try:
            async with driver.session() as session:
                # Execute a simple query to verify connectivity
                result = await session.run("RETURN 1 AS n")
                await result.single()

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug("neo4j_health_check_passed", latency_ms=latency_ms)
            return ComponentHealth(status=ComponentStatus.UP, latency_ms=latency_ms)
        finally:
            await driver.close()
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning("neo4j_health_check_failed", error=error_msg)
        return ComponentHealth(status=ComponentStatus.DOWN, latency_ms=latency_ms, error=error_msg)


async def _check_slack_health(settings: Settings) -> ComponentHealth:
    """Check Slack API connectivity.

    Args:
        settings: Application settings containing Slack configuration.

    Returns:
        ComponentHealth with the status of the Slack connection.
    """
    from slack_sdk.web.async_client import AsyncWebClient

    start_time = time.perf_counter()
    try:
        client = AsyncWebClient(token=settings.slack.bot_token.get_secret_value())
        response = await client.auth_test()

        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.get("ok"):
            logger.debug("slack_health_check_passed", latency_ms=latency_ms)
            return ComponentHealth(status=ComponentStatus.UP, latency_ms=latency_ms)
        else:
            error_msg = response.get("error", "Unknown error")
            logger.warning("slack_health_check_failed", error=error_msg)
            return ComponentHealth(
                status=ComponentStatus.DOWN, latency_ms=latency_ms, error=error_msg
            )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning("slack_health_check_failed", error=error_msg)
        return ComponentHealth(status=ComponentStatus.DOWN, latency_ms=latency_ms, error=error_msg)


async def _check_gemini_health(settings: Settings) -> ComponentHealth:
    """Check Gemini API connectivity.

    Args:
        settings: Application settings containing Gemini configuration.

    Returns:
        ComponentHealth with the status of the Gemini connection.
    """
    start_time = time.perf_counter()
    try:
        from google import genai

        client = genai.Client(api_key=settings.gemini.google_api_key.get_secret_value())

        # Use the list_models API to check connectivity
        # This is a lightweight call that verifies API key validity
        models = client.models.list()
        # Iterate once to verify the response
        next(iter(models), None)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.debug("gemini_health_check_passed", latency_ms=latency_ms)
        return ComponentHealth(status=ComponentStatus.UP, latency_ms=latency_ms)
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning("gemini_health_check_failed", error=error_msg)
        return ComponentHealth(status=ComponentStatus.DOWN, latency_ms=latency_ms, error=error_msg)


def _determine_overall_status(
    components: dict[str, ComponentHealth],
) -> OverallStatus:
    """Determine the overall health status based on component statuses.

    Args:
        components: Dictionary of component health statuses.

    Returns:
        The overall health status.

    Logic:
        - HEALTHY: All components are UP
        - DEGRADED: At least one component is DEGRADED or DOWN, but not all
        - UNHEALTHY: All components are DOWN
    """
    statuses = [c.status for c in components.values()]

    if all(s == ComponentStatus.UP for s in statuses):
        return OverallStatus.HEALTHY

    if all(s == ComponentStatus.DOWN for s in statuses):
        return OverallStatus.UNHEALTHY

    return OverallStatus.DEGRADED


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the health status of all system components.",
    responses={
        200: {
            "description": "Health status retrieved successfully",
            "model": HealthResponse,
        }
    },
)
async def health_check(
    settings: Settings = Depends(get_settings),  # noqa: B008 - Dependency injection
) -> HealthResponse:
    """Check the health of all system components.

    This endpoint checks the connectivity and status of:
    - Neo4j database
    - Slack API
    - Gemini API

    Returns:
        HealthResponse containing the overall status and individual component statuses.
    """
    logger.info("health_check_requested")

    # Run all health checks
    neo4j_health = await _check_neo4j_health(settings)
    slack_health = await _check_slack_health(settings)
    gemini_health = await _check_gemini_health(settings)

    components = {
        "neo4j": neo4j_health,
        "slack": slack_health,
        "gemini": gemini_health,
    }

    overall_status = _determine_overall_status(components)

    logger.info(
        "health_check_completed",
        overall_status=overall_status.value,
        neo4j_status=neo4j_health.status.value,
        slack_status=slack_health.status.value,
        gemini_status=gemini_health.status.value,
    )

    return HealthResponse(
        status=overall_status,
        components=components,
        version=settings.app.app_version,
    )


# Export public API
__all__ = [
    "router",
    "HealthResponse",
    "ComponentHealth",
    "ComponentStatus",
    "OverallStatus",
]
