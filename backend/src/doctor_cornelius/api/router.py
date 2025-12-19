"""Main API router for Doctor Cornelius.

This module provides the main API router that includes all endpoint routers
and configures the FastAPI application.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from doctor_cornelius.api.endpoints import health, ingest, search
from doctor_cornelius.config import Settings, get_settings

logger = structlog.get_logger(__name__)


def create_api_router() -> APIRouter:
    """Create the main API router with all endpoint routers included.

    Returns:
        APIRouter configured with all endpoint routers.
    """
    api_router = APIRouter()

    # Include endpoint routers
    api_router.include_router(health.router)
    api_router.include_router(search.router)
    api_router.include_router(ingest.router)

    logger.info(
        "api_router_created",
        routes=[
            "/health",
            "/search",
            "/search/episodes/{group_id}",
            "/ingest/backfill",
            "/ingest/trigger-daily",
        ],
    )

    return api_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. If not provided, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Doctor Cornelius API",
        description=(
            "Temporal Knowledge Graph API for team memory. "
            "Doctor Cornelius captures, indexes, and surfaces organizational "
            "knowledge from Slack conversations and other sources."
        ),
        version=settings.app.app_version,
        debug=settings.app.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the main API router
    api_router = create_api_router()
    app.include_router(api_router, prefix="/api/v1")

    # Also mount health check at root level for easier access
    app.include_router(health.router, tags=["health"])

    @app.on_event("startup")
    async def startup_event() -> None:
        """Handle application startup."""
        logger.info(
            "application_starting",
            app_name=settings.app.app_name,
            version=settings.app.app_version,
            debug=settings.app.debug,
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Handle application shutdown."""
        logger.info("application_shutting_down")

    logger.info(
        "fastapi_app_created",
        title=app.title,
        version=app.version,
        debug=app.debug,
    )

    return app


# Lazy application instance - created on first access
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get the FastAPI application instance (lazy initialization).

    Returns:
        FastAPI application instance.
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app


# Export public API
__all__ = [
    "get_app",
    "create_app",
    "create_api_router",
]
