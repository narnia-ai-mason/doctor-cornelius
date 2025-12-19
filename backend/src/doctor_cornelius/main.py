"""FastAPI main entry point for Doctor Cornelius.

This module provides the main application entry point that orchestrates:
- FastAPI REST API
- Slack Bot (Socket Mode)
- APScheduler for daily collection jobs

Run modes:
- api: API server only
- bot: Slack bot only
- scheduler: Scheduler only
- all: All services (default)

Usage:
    # Run all services (default)
    python -m doctor_cornelius.main

    # Run specific service
    RUN_MODE=api python -m doctor_cornelius.main
    python -m doctor_cornelius.main --mode bot

    # Using uvicorn directly (API only)
    uvicorn doctor_cornelius.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
import uvicorn
from fastapi import FastAPI

from doctor_cornelius.agent import close_agent_manager
from doctor_cornelius.api.router import create_api_router
from doctor_cornelius.bot.app import (
    create_app as create_slack_app,
)
from doctor_cornelius.bot.app import (
    start_socket_mode,
)
from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.scheduler.jobs import (
    GracefulShutdown,
    create_scheduler,
)

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from slack_bolt.async_app import AsyncApp


class RunMode(str, Enum):
    """Available run modes for the application."""

    API = "api"
    BOT = "bot"
    SCHEDULER = "scheduler"
    ALL = "all"


# Global instances for lifespan management
_scheduler: AsyncIOScheduler | None = None
_slack_app: AsyncApp | None = None
_bot_task: asyncio.Task[None] | None = None
_shutdown_handler: GracefulShutdown | None = None


def configure_logging(settings: Settings) -> None:
    """Configure structlog for structured logging.

    Sets up JSON logging for production environments and
    colorful console logging for development.

    Args:
        settings: Application settings containing log configuration.
    """
    # Determine processors based on log format
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.app.log_format == "json":
        # Production: JSON format
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Console format with colors
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.app.log_level),
    )

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "apscheduler"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_run_mode() -> RunMode:
    """Determine the run mode from CLI args or environment variable.

    Checks command line arguments first, then falls back to RUN_MODE
    environment variable, defaulting to 'all' if neither is set.

    Returns:
        The determined RunMode.
    """
    # Check CLI args first (when running as module)
    parser = argparse.ArgumentParser(description="Doctor Cornelius Application")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=[m.value for m in RunMode],
        default=None,
        help="Run mode: api, bot, scheduler, or all (default: all)",
    )
    args, _ = parser.parse_known_args()

    if args.mode:
        return RunMode(args.mode)

    # Fall back to environment variable
    env_mode = os.environ.get("RUN_MODE", "all").lower()
    try:
        return RunMode(env_mode)
    except ValueError:
        return RunMode.ALL


async def start_scheduler(settings: Settings) -> AsyncIOScheduler:
    """Start the APScheduler for scheduled jobs.

    Creates and starts the scheduler with the daily collection job.

    Args:
        settings: Application settings.

    Returns:
        The started AsyncIOScheduler instance.
    """
    global _scheduler, _shutdown_handler

    log = structlog.get_logger(__name__).bind(component="scheduler")
    log.info("starting_scheduler")

    _scheduler = create_scheduler(settings=settings)
    _shutdown_handler = _scheduler._doctor_cornelius_shutdown_handler  # type: ignore[attr-defined]

    _scheduler.start()
    log.info(
        "scheduler_started",
        jobs=[job.id for job in _scheduler.get_jobs()],
    )

    return _scheduler


async def stop_scheduler() -> None:
    """Stop the scheduler gracefully.

    Shuts down the scheduler and waits for running jobs to complete.
    """
    global _scheduler, _shutdown_handler

    log = structlog.get_logger(__name__).bind(component="scheduler")

    if _shutdown_handler:
        _shutdown_handler.request_shutdown()

    if _scheduler:
        log.info("stopping_scheduler")
        _scheduler.shutdown(wait=True)
        _scheduler = None
        log.info("scheduler_stopped")


async def start_slack_bot(settings: Settings) -> asyncio.Task[None]:
    """Start the Slack bot in Socket Mode as a background task.

    Creates the Slack Bolt app and starts it in Socket Mode.
    Runs as a background asyncio task.

    Args:
        settings: Application settings.

    Returns:
        The background task running the bot.
    """
    global _slack_app, _bot_task

    log = structlog.get_logger(__name__).bind(component="slack_bot")
    log.info("starting_slack_bot")

    _slack_app = create_slack_app(settings)

    async def run_bot() -> None:
        """Run the Slack bot Socket Mode."""
        try:
            await start_socket_mode(_slack_app, settings)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            log.info("slack_bot_cancelled")
        except Exception as e:
            log.error(
                "slack_bot_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )

    _bot_task = asyncio.create_task(run_bot())
    log.info("slack_bot_started")

    return _bot_task


async def stop_slack_bot() -> None:
    """Stop the Slack bot gracefully.

    Cancels the bot task and closes the Graphiti manager.
    """
    global _slack_app, _bot_task

    log = structlog.get_logger(__name__).bind(component="slack_bot")

    if _bot_task and not _bot_task.done():
        log.info("stopping_slack_bot")
        _bot_task.cancel()
        try:
            await asyncio.wait_for(_bot_task, timeout=5.0)
        except TimeoutError:
            log.warning("slack_bot_stop_timeout")
        except asyncio.CancelledError:
            pass
        _bot_task = None

    # Close the agent manager used by the bot
    await close_agent_manager()
    _slack_app = None
    log.info("slack_bot_stopped")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application.

    Handles startup and shutdown of all services based on run mode:
    - Starts scheduler on startup (if mode is 'scheduler' or 'all')
    - Starts Slack bot on startup (if mode is 'bot' or 'all')
    - Gracefully shuts down all services on shutdown

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup is complete.
    """
    settings = get_settings()
    log = structlog.get_logger(__name__).bind(component="lifespan")

    # Get the run mode from app state or determine it
    run_mode = getattr(app.state, "run_mode", get_run_mode())

    log.info(
        "application_starting",
        run_mode=run_mode.value,
        app_name=settings.app.app_name,
        version=settings.app.app_version,
        debug=settings.app.debug,
    )

    # Start services based on run mode
    try:
        if run_mode in (RunMode.SCHEDULER, RunMode.ALL):
            await start_scheduler(settings)

        if run_mode in (RunMode.BOT, RunMode.ALL):
            await start_slack_bot(settings)

        log.info(
            "application_started",
            run_mode=run_mode.value,
            scheduler_running=_scheduler is not None,
            bot_running=_bot_task is not None,
        )

        yield

    finally:
        # Shutdown services
        log.info("application_shutting_down")

        # Stop services in reverse order
        if run_mode in (RunMode.BOT, RunMode.ALL):
            await stop_slack_bot()

        if run_mode in (RunMode.SCHEDULER, RunMode.ALL):
            await stop_scheduler()

        log.info("application_shutdown_complete")


def create_app_with_lifespan(
    settings: Settings | None = None,
    run_mode: RunMode | None = None,
) -> FastAPI:
    """Create the FastAPI application with lifespan management.

    Creates a FastAPI application with:
    - API router mounted at /api/v1
    - Health check at root level
    - Lifespan handler for startup/shutdown of scheduler and bot
    - CORS middleware

    Args:
        settings: Application settings. Defaults to loading from environment.
        run_mode: Run mode for the application. Defaults to auto-detection.

    Returns:
        Configured FastAPI application with lifespan management.
    """
    settings = settings or get_settings()
    run_mode = run_mode or get_run_mode()

    # Configure logging first
    configure_logging(settings)

    log = structlog.get_logger(__name__)
    log.info(
        "creating_application",
        run_mode=run_mode.value,
        api_enabled=run_mode in (RunMode.API, RunMode.ALL),
        bot_enabled=run_mode in (RunMode.BOT, RunMode.ALL),
        scheduler_enabled=run_mode in (RunMode.SCHEDULER, RunMode.ALL),
    )

    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Doctor Cornelius API",
        description=(
            "Temporal Knowledge Graph API for team memory. "
            "Doctor Cornelius captures, indexes, and surfaces organizational "
            "knowledge from Slack conversations and other sources."
        ),
        version=settings.app.app_version,
        debug=settings.app.debug,
        docs_url="/docs" if run_mode in (RunMode.API, RunMode.ALL) else None,
        redoc_url="/redoc" if run_mode in (RunMode.API, RunMode.ALL) else None,
        openapi_url="/openapi.json" if run_mode in (RunMode.API, RunMode.ALL) else None,
        lifespan=lifespan,
    )

    # Store run mode and settings in app state
    app.state.run_mode = run_mode
    app.state.settings = settings

    # Only add API routes if in API or ALL mode
    if run_mode in (RunMode.API, RunMode.ALL):
        from fastapi.middleware.cors import CORSMiddleware

        from doctor_cornelius.api.endpoints import health

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

        log.info("api_routes_configured")
    else:
        # Minimal health endpoint for non-API modes
        @app.get("/health")
        async def health_check() -> dict[str, str]:
            """Basic health check for non-API modes."""
            return {
                "status": "healthy",
                "mode": run_mode.value,
            }

    log.info(
        "application_created",
        title=app.title,
        version=app.version,
        debug=app.debug,
        run_mode=run_mode.value,
    )

    return app


def run_uvicorn(settings: Settings | None = None, run_mode: RunMode | None = None) -> None:
    """Run the application with Uvicorn.

    Starts the FastAPI application using Uvicorn with settings from config.

    Args:
        settings: Application settings. Defaults to loading from environment.
        run_mode: Run mode for the application. Defaults to auto-detection.
    """
    settings = settings or get_settings()
    run_mode = run_mode or get_run_mode()

    # Configure logging before starting
    configure_logging(settings)

    log = structlog.get_logger(__name__)
    log.info(
        "starting_uvicorn",
        host=settings.app.api_host,
        port=settings.app.api_port,
        run_mode=run_mode.value,
        debug=settings.app.debug,
    )

    # Set environment variable for the app factory to read
    os.environ["_DOCTOR_CORNELIUS_RUN_MODE"] = run_mode.value

    uvicorn.run(
        "doctor_cornelius.main:create_app_with_lifespan",
        host=settings.app.api_host,
        port=settings.app.api_port,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower(),
        factory=True,
    )


async def run_bot_only(settings: Settings | None = None) -> None:
    """Run only the Slack bot without the API server.

    Useful for deployments where the bot runs in a separate process.

    Args:
        settings: Application settings. Defaults to loading from environment.
    """
    settings = settings or get_settings()

    # Configure logging
    configure_logging(settings)

    log = structlog.get_logger(__name__).bind(component="bot_only")
    log.info("starting_bot_only_mode")

    try:
        slack_app = create_slack_app(settings)
        await start_socket_mode(slack_app, settings)
    except KeyboardInterrupt:
        log.info("bot_interrupted")
    except Exception as e:
        log.error(
            "bot_fatal_error",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise
    finally:
        await close_agent_manager()
        log.info("bot_stopped")


async def run_scheduler_only(settings: Settings | None = None) -> None:
    """Run only the scheduler without the API server.

    Useful for deployments where the scheduler runs in a separate process.

    Args:
        settings: Application settings. Defaults to loading from environment.
    """
    settings = settings or get_settings()

    # Configure logging
    configure_logging(settings)

    log = structlog.get_logger(__name__).bind(component="scheduler_only")
    log.info("starting_scheduler_only_mode")

    scheduler = create_scheduler(settings=settings)
    shutdown_handler: GracefulShutdown = scheduler._doctor_cornelius_shutdown_handler  # type: ignore[attr-defined]

    try:
        scheduler.start()
        log.info(
            "scheduler_started",
            jobs=[job.id for job in scheduler.get_jobs()],
        )

        # Keep running until shutdown is requested
        while not shutdown_handler.should_shutdown:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        log.info("scheduler_interrupted")
    finally:
        scheduler.shutdown(wait=True)
        log.info("scheduler_stopped")


def main() -> None:
    """Main entry point for the application.

    Determines run mode and starts the appropriate services:
    - api: Runs Uvicorn with API only
    - bot: Runs Slack bot only
    - scheduler: Runs scheduler only
    - all: Runs Uvicorn with all services (API + bot + scheduler)
    """
    settings = get_settings()
    run_mode = get_run_mode()

    # Configure logging first
    configure_logging(settings)

    log = structlog.get_logger(__name__)
    log.info(
        "main_starting",
        run_mode=run_mode.value,
        app_name=settings.app.app_name,
        version=settings.app.app_version,
    )

    try:
        if run_mode == RunMode.BOT:
            # Bot-only mode: run without Uvicorn
            asyncio.run(run_bot_only(settings))

        elif run_mode == RunMode.SCHEDULER:
            # Scheduler-only mode: run without Uvicorn
            asyncio.run(run_scheduler_only(settings))

        else:
            # API or ALL mode: run with Uvicorn
            run_uvicorn(settings, run_mode)

    except KeyboardInterrupt:
        log.info("application_interrupted")
    except Exception as e:
        log.error(
            "application_fatal_error",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        sys.exit(1)


# Lazy application instance for uvicorn direct usage
# (e.g., uvicorn doctor_cornelius.main:app)
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get the FastAPI application instance (lazy initialization).

    Returns:
        FastAPI application instance.
    """
    global _app
    if _app is None:
        _app = create_app_with_lifespan()
    return _app


# For uvicorn factory pattern: uvicorn doctor_cornelius.main:get_app --factory
# Or direct module attribute access will trigger lazy init
def __getattr__(name: str):
    """Lazy attribute access for module-level app."""
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Export public API
__all__ = [
    "get_app",
    "create_app_with_lifespan",
    "configure_logging",
    "get_run_mode",
    "main",
    "run_bot_only",
    "run_scheduler_only",
    "run_uvicorn",
    "RunMode",
]


if __name__ == "__main__":
    main()
