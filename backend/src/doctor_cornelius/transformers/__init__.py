"""Transformers for converting raw collected data into Episode format.

This package contains transformer implementations that convert raw data from
various collectors (Slack, Notion, GitHub, etc.) into standardized Episode
objects for ingestion into the knowledge graph.
"""

from doctor_cornelius.transformers.base import BaseTransformer, RawDataItemT
from doctor_cornelius.transformers.slack_transformer import (
    SYSTEM_MESSAGE_SUBTYPES,
    SlackTransformer,
    UserResolver,
)

__all__ = [
    "BaseTransformer",
    "RawDataItemT",
    "SlackTransformer",
    "SYSTEM_MESSAGE_SUBTYPES",
    "UserResolver",
]
