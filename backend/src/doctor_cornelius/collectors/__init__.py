"""Data collectors for Doctor Cornelius.

This package contains collectors for fetching data from various sources
like Slack, Notion, GitHub, and Jira.
"""

from doctor_cornelius.collectors.base import (
    BaseCollector,
    CollectionConfig,
    CollectionStats,
    ConfigT,
    DataSource,
    RawDataItem,
    SourceType,
)
from doctor_cornelius.collectors.slack_collector import SlackCollector

__all__ = [
    "BaseCollector",
    "CollectionConfig",
    "CollectionStats",
    "ConfigT",
    "DataSource",
    "RawDataItem",
    "SlackCollector",
    "SourceType",
]
