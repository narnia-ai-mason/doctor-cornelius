"""Ingestion endpoints for Doctor Cornelius API.

This module provides endpoints for triggering data collection and ingestion
into the temporal knowledge graph.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field

from doctor_cornelius.collectors.base import CollectionConfig
from doctor_cornelius.collectors.slack_collector import SlackCollector
from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.knowledge.graph_client import GraphitiClientManager
from doctor_cornelius.transformers.slack_transformer import SlackTransformer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class IngestionStatus(str, Enum):
    """Status of an ingestion job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BackfillRequest(BaseModel):
    """Request model for historical backfill ingestion."""

    channel_ids: list[str] | None = Field(
        default=None,
        description=(
            "Specific Slack channel IDs to backfill. "
            "If empty, backfills all accessible channels."
        ),
    )
    start_time: datetime | None = Field(
        default=None,
        description="Start time for historical collection (ISO 8601 format)",
    )
    end_time: datetime | None = Field(
        default=None,
        description="End time for historical collection (ISO 8601 format). Defaults to now.",
    )
    include_threads: bool = Field(
        default=True,
        description="Whether to include thread replies",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of messages to process per batch",
    )
    max_items: int | None = Field(
        default=None,
        ge=1,
        description="Maximum total items to collect (None for unlimited)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "channel_ids": ["C01234567", "C09876543"],
                    "start_time": "2024-01-01T00:00:00Z",
                    "end_time": "2024-06-01T00:00:00Z",
                    "include_threads": True,
                    "batch_size": 100,
                }
            ]
        }
    }


class DailyTriggerRequest(BaseModel):
    """Request model for manual daily collection trigger."""

    channel_ids: list[str] | None = Field(
        default=None,
        description="Specific Slack channel IDs to collect."
                    " If empty, collects from all accessible channels."
    )
    hours_back: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of hours back to collect (default: 24, max: 168/1 week)",
    )
    include_threads: bool = Field(
        default=True,
        description="Whether to include thread replies",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "channel_ids": None,
                    "hours_back": 24,
                    "include_threads": True,
                }
            ]
        }
    }


class IngestionJobResponse(BaseModel):
    """Response model for ingestion job initiation."""

    job_id: str = Field(description="Unique identifier for the ingestion job")
    status: IngestionStatus = Field(description="Current status of the job")
    message: str = Field(description="Status message")
    created_at: datetime = Field(description="When the job was created")
    config: dict[str, Any] = Field(description="Configuration used for this ingestion job")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "ing_abc123",
                    "status": "in_progress",
                    "message": "Backfill job started for 3 channels",
                    "created_at": "2024-01-15T10:30:00Z",
                    "config": {
                        "channel_ids": ["C01234567"],
                        "start_time": "2024-01-01T00:00:00Z",
                        "include_threads": True,
                    },
                }
            ]
        }
    }


class IngestionResult(BaseModel):
    """Result of a completed ingestion operation."""

    job_id: str = Field(description="Job identifier")
    status: IngestionStatus = Field(description="Final status")
    channels_processed: int = Field(description="Number of channels processed")
    messages_collected: int = Field(description="Number of messages collected")
    episodes_ingested: int = Field(description="Number of episodes ingested")
    errors: list[str] = Field(default_factory=list, description="List of errors")
    started_at: datetime = Field(description="When processing started")
    completed_at: datetime | None = Field(default=None, description="When processing completed")
    duration_seconds: float | None = Field(default=None, description="Total duration in seconds")


# -----------------------------------------------------------------------------
# Background Task Functions
# -----------------------------------------------------------------------------


async def _run_ingestion(
    job_id: str,
    config: CollectionConfig,
    settings: Settings,
) -> IngestionResult:
    """Run the ingestion process in the background.

    Args:
        job_id: Unique job identifier.
        config: Collection configuration.
        settings: Application settings.

    Returns:
        IngestionResult with the final status and statistics.
    """
    log = logger.bind(job_id=job_id)
    log.info("ingestion_job_started", config=config.model_dump())

    started_at = datetime.now(UTC)
    messages_collected = 0
    episodes_ingested = 0
    errors: list[str] = []

    try:
        # Initialize components
        collector = SlackCollector(settings=settings)
        transformer = SlackTransformer(
            user_resolver=collector._get_user_name,
        )

        async with GraphitiClientManager(settings=settings) as graph_client:
            # Collect messages
            batch: list[Any] = []
            async for raw_item in collector.collect(config):
                messages_collected += 1

                # Transform to episode
                episode = await transformer.transform(raw_item)
                if episode is None:
                    continue

                batch.append(episode)

                # Process batch when it reaches the configured size
                if len(batch) >= settings.collector.batch_size:
                    try:
                        result = await graph_client.ingest_episodes_batch(batch)
                        episodes_ingested += result.get("episode_count", 0)
                        log.debug(
                            "batch_ingested",
                            batch_size=len(batch),
                            episodes=result.get("episode_count", 0),
                        )
                    except Exception as e:
                        error_msg = f"Batch ingestion failed: {str(e)}"
                        errors.append(error_msg)
                        log.error("batch_ingestion_failed", error=str(e))

                    batch = []

                    # Delay between batches to avoid rate limiting
                    await asyncio.sleep(settings.collector.batch_delay_seconds)

            # Process remaining items
            if batch:
                try:
                    result = await graph_client.ingest_episodes_batch(batch)
                    episodes_ingested += result.get("episode_count", 0)
                except Exception as e:
                    error_msg = f"Final batch ingestion failed: {str(e)}"
                    errors.append(error_msg)
                    log.error("final_batch_ingestion_failed", error=str(e))

        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()

        final_status = IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED

        log.info(
            "ingestion_job_completed",
            status=final_status.value,
            messages_collected=messages_collected,
            episodes_ingested=episodes_ingested,
            error_count=len(errors),
            duration_seconds=duration,
        )

        return IngestionResult(
            job_id=job_id,
            status=final_status,
            channels_processed=collector.stats.sources_processed,
            messages_collected=messages_collected,
            episodes_ingested=episodes_ingested,
            errors=errors,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    except Exception as e:
        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()
        error_msg = f"Ingestion job failed: {type(e).__name__}: {str(e)}"
        errors.append(error_msg)

        log.error(
            "ingestion_job_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )

        return IngestionResult(
            job_id=job_id,
            status=IngestionStatus.FAILED,
            channels_processed=0,
            messages_collected=messages_collected,
            episodes_ingested=episodes_ingested,
            errors=errors,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post(
    "/backfill",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Historical Backfill",
    description="Trigger a historical data collection and ingestion job.",
    responses={
        202: {"description": "Backfill job accepted and started"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service unavailable"},
    },
)
async def trigger_backfill(
    request: BackfillRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),  # noqa: B008 - Dependency injection
) -> IngestionJobResponse:
    """Trigger historical data collection from Slack channels.

    This endpoint initiates a background job to collect and ingest
    historical Slack messages into the knowledge graph.

    The job runs asynchronously and returns immediately with a job ID.

    Args:
        request: Backfill configuration.
        background_tasks: FastAPI background tasks handler.
        settings: Application settings.

    Returns:
        IngestionJobResponse with job ID and initial status.
    """
    job_id = f"backfill_{uuid4().hex[:12]}"
    created_at = datetime.now(UTC)

    logger.info(
        "backfill_request_received",
        job_id=job_id,
        channel_ids=request.channel_ids,
        start_time=request.start_time,
        end_time=request.end_time,
    )

    # Build collection config
    config = CollectionConfig(
        source_ids=request.channel_ids or [],
        start_time=request.start_time,
        end_time=request.end_time or datetime.now(UTC),
        include_threads=request.include_threads,
        include_replies=request.include_threads,
        batch_size=request.batch_size,
        max_items=request.max_items,
    )

    # Schedule background task
    background_tasks.add_task(_run_ingestion, job_id, config, settings)

    channel_count = len(request.channel_ids) if request.channel_ids else "all"
    message = f"Backfill job started for {channel_count} channels"

    return IngestionJobResponse(
        job_id=job_id,
        status=IngestionStatus.IN_PROGRESS,
        message=message,
        created_at=created_at,
        config={
            "channel_ids": request.channel_ids,
            "start_time": request.start_time.isoformat() if request.start_time else None,
            "end_time": request.end_time.isoformat() if request.end_time else None,
            "include_threads": request.include_threads,
            "batch_size": request.batch_size,
            "max_items": request.max_items,
        },
    )


@router.post(
    "/trigger-daily",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Daily Collection",
    description="Manually trigger the daily data collection job.",
    responses={
        202: {"description": "Daily collection job accepted and started"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service unavailable"},
    },
)
async def trigger_daily_collection(
    request: DailyTriggerRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),  # noqa: B008 - Dependency injection
) -> IngestionJobResponse:
    """Manually trigger a daily data collection job.

    This endpoint initiates a background job to collect and ingest
    recent Slack messages (default: last 24 hours) into the knowledge graph.

    The job runs asynchronously and returns immediately with a job ID.

    Args:
        request: Daily collection configuration.
        background_tasks: FastAPI background tasks handler.
        settings: Application settings.

    Returns:
        IngestionJobResponse with job ID and initial status.
    """
    job_id = f"daily_{uuid4().hex[:12]}"
    created_at = datetime.now(UTC)

    logger.info(
        "daily_trigger_request_received",
        job_id=job_id,
        channel_ids=request.channel_ids,
        hours_back=request.hours_back,
    )

    # Calculate time range
    end_time = datetime.now(UTC)
    start_time = datetime.fromtimestamp(
        end_time.timestamp() - (request.hours_back * 3600),
        tz=UTC,
    )

    # Build collection config
    config = CollectionConfig(
        source_ids=request.channel_ids or [],
        start_time=start_time,
        end_time=end_time,
        include_threads=request.include_threads,
        include_replies=request.include_threads,
        batch_size=settings.collector.batch_size,
    )

    # Schedule background task
    background_tasks.add_task(_run_ingestion, job_id, config, settings)

    channel_count = len(request.channel_ids) if request.channel_ids else "all"
    message = (
        f"Daily collection job started for {channel_count} channels "
        f"(last {request.hours_back} hours)"
    )

    return IngestionJobResponse(
        job_id=job_id,
        status=IngestionStatus.IN_PROGRESS,
        message=message,
        created_at=created_at,
        config={
            "channel_ids": request.channel_ids,
            "hours_back": request.hours_back,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "include_threads": request.include_threads,
        },
    )


# Export public API
__all__ = [
    "router",
    "BackfillRequest",
    "DailyTriggerRequest",
    "IngestionJobResponse",
    "IngestionResult",
    "IngestionStatus",
]
