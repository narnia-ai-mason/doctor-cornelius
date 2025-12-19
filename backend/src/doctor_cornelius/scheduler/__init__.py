"""Scheduler module for Doctor Cornelius automated data collection.

This module provides APScheduler-based job scheduling for daily Slack
data collection and historical backfill operations.

Usage:
    from doctor_cornelius.scheduler import create_scheduler, run_daily_collection

    # Start automated scheduling
    scheduler = create_scheduler()
    scheduler.start()

    # Or run manual collection
    result = await run_daily_collection()
"""

from doctor_cornelius.scheduler.jobs import (
    GracefulShutdown,
    JobCheckpoint,
    create_scheduler,
    run_backfill,
    run_daily_collection,
    run_scheduler_async,
)

__all__ = [
    "create_scheduler",
    "run_daily_collection",
    "run_backfill",
    "JobCheckpoint",
    "GracefulShutdown",
    "run_scheduler_async",
]
