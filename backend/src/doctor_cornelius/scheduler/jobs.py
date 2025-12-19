"""Scheduler jobs for Doctor Cornelius daily data collection.

This module implements APScheduler-based scheduling for automated data collection
from Slack and ingestion into the Graphiti knowledge graph.

Key features:
- Daily collection job running at midnight
- Historical backfill functionality for date ranges
- Graceful shutdown handling (SIGTERM/SIGINT)
- Checkpoint saving for failure recovery
- Structured logging with job context

Usage:
    # Start the scheduler
    scheduler = create_scheduler()
    scheduler.start()

    # Or run a backfill
    await run_backfill(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31)
    )
"""

import asyncio
import json
import signal
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from doctor_cornelius.collectors.base import CollectionConfig
from doctor_cornelius.collectors.slack_collector import SlackCollector
from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.knowledge.graph_client import GraphitiClientManager
from doctor_cornelius.schemas.episode import Episode
from doctor_cornelius.transformers.slack_transformer import SlackTransformer

logger = structlog.get_logger(__name__)

# Default checkpoint file location
DEFAULT_CHECKPOINT_PATH = Path("/tmp/doctor_cornelius_checkpoint.json")


class JobCheckpoint:
    """Checkpoint manager for job failure recovery.

    Saves and loads checkpoint data to allow resumption of interrupted
    collection jobs. Checkpoints track:
    - Job type (daily or backfill)
    - Start/end dates
    - Channels processed
    - Last processed timestamp
    - Error information

    Usage:
        checkpoint = JobCheckpoint(checkpoint_path)
        checkpoint.save(job_type="daily", channels_processed=["C123", "C456"])
        data = checkpoint.load()
    """

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_path: Path to save checkpoint data.
                Defaults to /tmp/doctor_cornelius_checkpoint.json
        """
        self._path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self._log = logger.bind(component="checkpoint", path=str(self._path))

    def save(
        self,
        job_type: str,
        start_date: date | None = None,
        end_date: date | None = None,
        channels_processed: list[str] | None = None,
        last_message_ts: str | None = None,
        error: str | None = None,
        episodes_ingested: int = 0,
    ) -> None:
        """Save checkpoint data to disk.

        Args:
            job_type: Type of job ("daily" or "backfill").
            start_date: Collection start date.
            end_date: Collection end date.
            channels_processed: List of channel IDs already processed.
            last_message_ts: Timestamp of last processed message.
            error: Error message if job failed.
            episodes_ingested: Number of episodes successfully ingested.
        """
        checkpoint_data = {
            "job_type": job_type,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "channels_processed": channels_processed or [],
            "last_message_ts": last_message_ts,
            "error": error,
            "episodes_ingested": episodes_ingested,
            "saved_at": datetime.now(UTC).isoformat(),
        }

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(checkpoint_data, indent=2))
            self._log.debug(
                "checkpoint_saved",
                job_type=job_type,
                episodes_ingested=episodes_ingested,
            )
        except Exception as e:
            self._log.error(
                "checkpoint_save_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )

    def load(self) -> dict[str, Any] | None:
        """Load checkpoint data from disk.

        Returns:
            Checkpoint data dictionary, or None if no checkpoint exists.
        """
        try:
            if not self._path.exists():
                return None
            data = json.loads(self._path.read_text())
            self._log.info(
                "checkpoint_loaded",
                job_type=data.get("job_type"),
                saved_at=data.get("saved_at"),
            )
            return data
        except Exception as e:
            self._log.error(
                "checkpoint_load_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return None

    def clear(self) -> None:
        """Remove the checkpoint file."""
        try:
            if self._path.exists():
                self._path.unlink()
                self._log.debug("checkpoint_cleared")
        except Exception as e:
            self._log.warning(
                "checkpoint_clear_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )


class GracefulShutdown:
    """Handler for graceful shutdown on SIGTERM/SIGINT.

    Manages the shutdown state and allows running jobs to complete
    their current batch before exiting.

    Usage:
        shutdown_handler = GracefulShutdown()
        shutdown_handler.register_signals()

        while not shutdown_handler.should_shutdown:
            # Process work
            pass
    """

    def __init__(self) -> None:
        """Initialize the shutdown handler."""
        self._shutdown_requested = False
        self._log = logger.bind(component="shutdown_handler")

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def request_shutdown(self) -> None:
        """Request a graceful shutdown."""
        self._shutdown_requested = True
        self._log.info("shutdown_requested")

    def register_signals(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT."""
        # Only register if we're in the main thread
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            self._log.info("signal_handlers_registered")
        except ValueError:
            # Signal handling only works in main thread
            self._log.warning("signal_handlers_not_registered_not_main_thread")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle received signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        signal_name = signal.Signals(signum).name
        self._log.info("signal_received", signal=signal_name)
        self.request_shutdown()


async def run_daily_collection(
    settings: Settings | None = None,
    checkpoint: JobCheckpoint | None = None,
    shutdown_handler: GracefulShutdown | None = None,
) -> dict[str, Any]:
    """Execute the daily Slack collection job.

    This job:
    1. Initializes the SlackCollector
    2. Collects messages from yesterday (last 24 hours)
    3. Transforms messages using SlackTransformer
    4. Ingests episodes into the knowledge graph via GraphitiClientManager

    Args:
        settings: Application settings. Defaults to loading from environment.
        checkpoint: Checkpoint manager for failure recovery.
        shutdown_handler: Handler for graceful shutdown.

    Returns:
        A dictionary containing job results:
            - status: "success", "partial", or "failed"
            - episodes_ingested: Number of episodes successfully ingested
            - channels_processed: Number of channels processed
            - errors: List of error messages
            - duration_seconds: Total job duration

    Raises:
        RuntimeError: If critical initialization fails.
    """
    settings = settings or get_settings()
    checkpoint = checkpoint or JobCheckpoint()
    shutdown_handler = shutdown_handler or GracefulShutdown()

    job_log = logger.bind(
        job_type="daily_collection",
        job_start=datetime.now(UTC).isoformat(),
    )
    job_log.info("daily_collection_started")

    start_time = datetime.now(UTC)
    result: dict[str, Any] = {
        "status": "failed",
        "episodes_ingested": 0,
        "channels_processed": 0,
        "errors": [],
        "duration_seconds": 0.0,
    }

    # Calculate yesterday's time range
    now = datetime.now(UTC)
    end_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time_collection = end_time - timedelta(days=1)

    job_log.info(
        "collection_time_range",
        start_time=start_time_collection.isoformat(),
        end_time=end_time.isoformat(),
    )

    # Save initial checkpoint
    checkpoint.save(
        job_type="daily",
        start_date=start_time_collection.date(),
        end_date=end_time.date(),
    )

    try:
        # Initialize collector
        collector = SlackCollector(settings=settings)

        # Validate Slack connection
        if not await collector.validate_connection():
            error_msg = "Failed to validate Slack connection"
            job_log.error(error_msg)
            result["errors"].append(error_msg)
            checkpoint.save(job_type="daily", error=error_msg)
            return result

        # Get the user name resolver for transformer
        async def user_resolver(user_id: str) -> str | None:
            return await collector._get_user_name(user_id)

        # Initialize transformer
        transformer = SlackTransformer(user_resolver=user_resolver)

        # Initialize graph client
        async with GraphitiClientManager(settings=settings) as graph_client:
            # Configure collection
            config = CollectionConfig(
                start_time=start_time_collection,
                end_time=end_time,
                include_threads=True,
                include_replies=True,
                batch_size=settings.collector.batch_size,
            )

            # Collect and process messages
            episodes_batch: list[Episode] = []
            channels_processed: set[str] = set()
            batch_size = settings.collector.batch_size

            async for raw_item in collector.collect(config):
                # Check for shutdown request
                if shutdown_handler.should_shutdown:
                    job_log.warning("shutdown_requested_during_collection")
                    break

                # Transform to episode
                episode = await transformer.transform(raw_item)
                if episode is None:
                    continue

                episodes_batch.append(episode)
                channels_processed.add(raw_item.group_id)

                # Ingest batch when it reaches the configured size
                if len(episodes_batch) >= batch_size:
                    try:
                        batch_result = await graph_client.ingest_episodes_batch(episodes_batch)
                        result["episodes_ingested"] += batch_result.get("episode_count", 0)
                        job_log.info(
                            "batch_ingested",
                            batch_size=len(episodes_batch),
                            total_ingested=result["episodes_ingested"],
                        )
                    except Exception as e:
                        error_msg = f"Batch ingestion failed: {e}"
                        job_log.error(
                            "batch_ingestion_failed",
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                        result["errors"].append(error_msg)

                    episodes_batch = []

                    # Update checkpoint after each batch
                    checkpoint.save(
                        job_type="daily",
                        start_date=start_time_collection.date(),
                        end_date=end_time.date(),
                        channels_processed=list(channels_processed),
                        episodes_ingested=result["episodes_ingested"],
                    )

                    # Apply batch delay
                    await asyncio.sleep(settings.collector.batch_delay_seconds)

            # Ingest remaining episodes
            if episodes_batch and not shutdown_handler.should_shutdown:
                try:
                    batch_result = await graph_client.ingest_episodes_batch(episodes_batch)
                    result["episodes_ingested"] += batch_result.get("episode_count", 0)
                    job_log.info(
                        "final_batch_ingested",
                        batch_size=len(episodes_batch),
                        total_ingested=result["episodes_ingested"],
                    )
                except Exception as e:
                    error_msg = f"Final batch ingestion failed: {e}"
                    job_log.error(
                        "final_batch_ingestion_failed",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    result["errors"].append(error_msg)

        result["channels_processed"] = len(channels_processed)

        # Determine final status
        if shutdown_handler.should_shutdown:
            result["status"] = "partial"
        elif result["errors"]:
            result["status"] = "partial" if result["episodes_ingested"] > 0 else "failed"
        else:
            result["status"] = "success"

    except Exception as e:
        error_msg = f"Daily collection failed: {e}"
        job_log.error(
            "daily_collection_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        result["errors"].append(error_msg)
        checkpoint.save(job_type="daily", error=error_msg)

    # Calculate duration
    result["duration_seconds"] = (datetime.now(UTC) - start_time).total_seconds()

    # Clear checkpoint on success
    if result["status"] == "success":
        checkpoint.clear()

    job_log.info(
        "daily_collection_completed",
        status=result["status"],
        episodes_ingested=result["episodes_ingested"],
        channels_processed=result["channels_processed"],
        error_count=len(result["errors"]),
        duration_seconds=result["duration_seconds"],
    )

    return result


async def run_backfill(
    start_date: date,
    end_date: date,
    channel_ids: list[str] | None = None,
    settings: Settings | None = None,
    checkpoint: JobCheckpoint | None = None,
    shutdown_handler: GracefulShutdown | None = None,
) -> dict[str, Any]:
    """Execute a historical backfill of Slack messages.

    This function collects messages from a specified date range and ingests
    them into the knowledge graph. Useful for initial setup or recovering
    missed data.

    Args:
        start_date: Start date for collection (inclusive).
        end_date: End date for collection (exclusive).
        channel_ids: Optional list of specific channel IDs to backfill.
            If None, backfills all accessible channels.
        settings: Application settings. Defaults to loading from environment.
        checkpoint: Checkpoint manager for failure recovery.
        shutdown_handler: Handler for graceful shutdown.

    Returns:
        A dictionary containing backfill results:
            - status: "success", "partial", or "failed"
            - episodes_ingested: Number of episodes successfully ingested
            - channels_processed: Number of channels processed
            - errors: List of error messages
            - duration_seconds: Total job duration
            - start_date: Requested start date
            - end_date: Requested end date

    Raises:
        ValueError: If start_date is after end_date.
        RuntimeError: If critical initialization fails.

    Example:
        >>> result = await run_backfill(
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 1, 31),
        ...     channel_ids=["C01234567"]
        ... )
        >>> print(f"Ingested {result['episodes_ingested']} episodes")
    """
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    settings = settings or get_settings()
    checkpoint = checkpoint or JobCheckpoint()
    shutdown_handler = shutdown_handler or GracefulShutdown()

    job_log = logger.bind(
        job_type="backfill",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        channel_ids=channel_ids,
        job_start=datetime.now(UTC).isoformat(),
    )
    job_log.info("backfill_started")

    job_start_time = datetime.now(UTC)
    result: dict[str, Any] = {
        "status": "failed",
        "episodes_ingested": 0,
        "channels_processed": 0,
        "errors": [],
        "duration_seconds": 0.0,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    # Convert dates to timezone-aware datetimes
    start_time = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
    end_time = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC)

    # Save initial checkpoint
    checkpoint.save(
        job_type="backfill",
        start_date=start_date,
        end_date=end_date,
    )

    try:
        # Initialize collector
        collector = SlackCollector(settings=settings)

        # Validate Slack connection
        if not await collector.validate_connection():
            error_msg = "Failed to validate Slack connection"
            job_log.error(error_msg)
            result["errors"].append(error_msg)
            checkpoint.save(job_type="backfill", error=error_msg)
            return result

        # Get the user name resolver for transformer
        async def user_resolver(user_id: str) -> str | None:
            return await collector._get_user_name(user_id)

        # Initialize transformer
        transformer = SlackTransformer(user_resolver=user_resolver)

        # Initialize graph client
        async with GraphitiClientManager(settings=settings) as graph_client:
            # Configure collection
            config = CollectionConfig(  # noqa: F841 - Unused variable
                source_ids=channel_ids or [],
                start_time=start_time,
                end_time=end_time,
                include_threads=True,
                include_replies=True,
                batch_size=settings.collector.batch_size,
            )

            # Process day by day for better checkpointing
            current_date = start_date
            channels_processed: set[str] = set()

            while current_date < end_date:
                if shutdown_handler.should_shutdown:
                    job_log.warning("shutdown_requested_during_backfill")
                    break

                day_start = datetime.combine(current_date, datetime.min.time(), tzinfo=UTC)
                day_end = day_start + timedelta(days=1)

                job_log.info(
                    "processing_day",
                    date=current_date.isoformat(),
                )

                # Update config for this day
                day_config = CollectionConfig(
                    source_ids=channel_ids or [],
                    start_time=day_start,
                    end_time=day_end,
                    include_threads=True,
                    include_replies=True,
                    batch_size=settings.collector.batch_size,
                )

                episodes_batch: list[Episode] = []
                batch_size = settings.collector.batch_size

                async for raw_item in collector.collect(day_config):
                    if shutdown_handler.should_shutdown:
                        break

                    # Transform to episode
                    episode = await transformer.transform(raw_item)
                    if episode is None:
                        continue

                    episodes_batch.append(episode)
                    channels_processed.add(raw_item.group_id)

                    # Ingest batch when it reaches the configured size
                    if len(episodes_batch) >= batch_size:
                        try:
                            batch_result = await graph_client.ingest_episodes_batch(episodes_batch)
                            result["episodes_ingested"] += batch_result.get("episode_count", 0)
                            job_log.info(
                                "batch_ingested",
                                batch_size=len(episodes_batch),
                                total_ingested=result["episodes_ingested"],
                            )
                        except Exception as e:
                            error_msg = f"Batch ingestion failed: {e}"
                            job_log.error(
                                "batch_ingestion_failed",
                                error_type=type(e).__name__,
                                error_message=str(e),
                            )
                            result["errors"].append(error_msg)

                        episodes_batch = []

                        # Update checkpoint after each batch
                        checkpoint.save(
                            job_type="backfill",
                            start_date=current_date,
                            end_date=end_date,
                            channels_processed=list(channels_processed),
                            episodes_ingested=result["episodes_ingested"],
                        )

                        # Apply batch delay
                        await asyncio.sleep(settings.collector.batch_delay_seconds)

                # Ingest remaining episodes for this day
                if episodes_batch and not shutdown_handler.should_shutdown:
                    try:
                        batch_result = await graph_client.ingest_episodes_batch(episodes_batch)
                        result["episodes_ingested"] += batch_result.get("episode_count", 0)
                    except Exception as e:
                        error_msg = f"Day batch ingestion failed: {e}"
                        job_log.error(
                            "day_batch_ingestion_failed",
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                        result["errors"].append(error_msg)

                current_date += timedelta(days=1)

        result["channels_processed"] = len(channels_processed)

        # Determine final status
        if shutdown_handler.should_shutdown:
            result["status"] = "partial"
        elif result["errors"]:
            result["status"] = "partial" if result["episodes_ingested"] > 0 else "failed"
        else:
            result["status"] = "success"

    except Exception as e:
        error_msg = f"Backfill failed: {e}"
        job_log.error(
            "backfill_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        result["errors"].append(error_msg)
        checkpoint.save(
            job_type="backfill",
            start_date=start_date,
            end_date=end_date,
            error=error_msg,
        )

    # Calculate duration
    result["duration_seconds"] = (datetime.now(UTC) - job_start_time).total_seconds()

    # Clear checkpoint on success
    if result["status"] == "success":
        checkpoint.clear()

    job_log.info(
        "backfill_completed",
        status=result["status"],
        episodes_ingested=result["episodes_ingested"],
        channels_processed=result["channels_processed"],
        error_count=len(result["errors"]),
        duration_seconds=result["duration_seconds"],
    )

    return result


def create_scheduler(
    settings: Settings | None = None,
    start_paused: bool = False,
) -> AsyncIOScheduler:
    """Create and configure the APScheduler instance.

    Creates an AsyncIOScheduler with the daily collection job configured
    to run at midnight UTC. The scheduler can be started immediately or
    paused for testing.

    Args:
        settings: Application settings. Defaults to loading from environment.
        start_paused: If True, create scheduler but don't start it.

    Returns:
        Configured AsyncIOScheduler instance.

    Example:
        >>> scheduler = create_scheduler()
        >>> scheduler.start()
        >>> # ... application runs ...
        >>> scheduler.shutdown()

    Note:
        The scheduler uses AsyncIOScheduler which is designed for
        asyncio-based applications. It integrates with the running
        event loop.
    """
    settings = settings or get_settings()

    scheduler_log = logger.bind(component="scheduler")
    scheduler_log.info("creating_scheduler")

    # Create shutdown handler for signal management
    shutdown_handler = GracefulShutdown()
    shutdown_handler.register_signals()

    # Create checkpoint manager
    checkpoint = JobCheckpoint()

    # Create the scheduler
    scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,  # Combine missed jobs into one
            "max_instances": 1,  # Only one instance of each job at a time
            "misfire_grace_time": 3600,  # Allow 1 hour grace for misfires
        },
        timezone="UTC",
    )

    # Wrapper to pass dependencies to the job
    async def daily_collection_wrapper() -> dict[str, Any]:
        """Wrapper for daily collection job with dependencies."""
        return await run_daily_collection(
            settings=settings,
            checkpoint=checkpoint,
            shutdown_handler=shutdown_handler,
        )

    # Add daily collection job - runs at midnight UTC
    scheduler.add_job(
        daily_collection_wrapper,
        trigger=CronTrigger(hour=0, minute=0, timezone="UTC"),
        id="daily_slack_collection",
        name="Daily Slack Collection",
        replace_existing=True,
    )

    scheduler_log.info(
        "scheduler_created",
        jobs=[job.id for job in scheduler.get_jobs()],
    )

    # Add shutdown callback
    def on_shutdown(event: Any) -> None:
        scheduler_log.info("scheduler_shutdown_event")
        shutdown_handler.request_shutdown()

    # Store references for access
    scheduler._doctor_cornelius_shutdown_handler = shutdown_handler  # type: ignore
    scheduler._doctor_cornelius_checkpoint = checkpoint  # type: ignore
    scheduler._doctor_cornelius_settings = settings  # type: ignore

    return scheduler


async def run_scheduler_async(settings: Settings | None = None) -> None:
    """Run the scheduler in the current event loop.

    This function creates and runs the scheduler, handling graceful
    shutdown on SIGTERM/SIGINT signals.

    Args:
        settings: Application settings. Defaults to loading from environment.

    Example:
        >>> asyncio.run(run_scheduler_async())
    """
    settings = settings or get_settings()
    scheduler_log = logger.bind(component="scheduler_runner")

    scheduler = create_scheduler(settings=settings)
    shutdown_handler = scheduler._doctor_cornelius_shutdown_handler  # type: ignore

    try:
        scheduler.start()
        scheduler_log.info("scheduler_started")

        # Keep running until shutdown is requested
        while not shutdown_handler.should_shutdown:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        scheduler_log.info("scheduler_interrupted")
    finally:
        scheduler.shutdown(wait=True)
        scheduler_log.info("scheduler_stopped")


# Export public API
__all__ = [
    "create_scheduler",
    "run_daily_collection",
    "run_backfill",
    "JobCheckpoint",
    "GracefulShutdown",
    "run_scheduler_async",
]
