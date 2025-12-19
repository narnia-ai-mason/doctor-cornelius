"""Standardized Episode schema for knowledge graph ingestion.

This module defines the Episode schema that serves as the standard format
for all data sources (Slack, Notion, GitHub, etc.) before ingestion into
the Graphiti temporal knowledge graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EpisodeSource(str, Enum):
    """Supported data sources for episodes."""

    SLACK = "slack"
    NOTION = "notion"
    GITHUB = "github"
    JIRA = "jira"
    MANUAL = "manual"


class GraphitiSourceType(str, Enum):
    """Graphiti EpisodeType values for add_episode() source parameter.

    These map to graphiti_core.nodes.EpisodeType enum values.
    """

    TEXT = "text"
    MESSAGE = "message"
    JSON = "json"


class Episode(BaseModel):
    """Standardized episode format for knowledge graph ingestion.

    An episode represents a discrete unit of information from any data source
    that can be ingested into the Graphiti temporal knowledge graph. This schema
    normalizes data from various sources (Slack messages, Notion pages, etc.)
    into a consistent format.

    Attributes:
        name: Episode name/title (e.g., "Message from @alice in #engineering").
        body: The main content of the episode.
        source: The data source identifier (e.g., "slack", "notion").
        reference_time: When the event/content occurred (timezone-aware).
        group_id: Logical grouping identifier (e.g., Slack channel_id).
        metadata: Additional source-specific data for context and traceability.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Episode name/title for identification",
    )
    body: str = Field(
        ...,
        min_length=1,
        description="The main content of the episode",
    )
    source: EpisodeSource = Field(
        ...,
        description="The data source this episode originated from",
    )
    reference_time: datetime = Field(
        ...,
        description="When the event/content occurred (must be timezone-aware)",
    )
    group_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Logical grouping identifier (e.g., channel_id for Slack)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific data",
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Message from @alice in #engineering",
                    "body": "We should implement the new caching layer this sprint.",
                    "source": "slack",
                    "reference_time": "2024-01-15T10:30:00Z",
                    "group_id": "C01234567",
                    "metadata": {
                        "message_ts": "1705315800.000100",
                        "user_id": "U01234567",
                        "thread_ts": None,
                        "channel_name": "engineering",
                    },
                }
            ]
        },
    }

    @field_validator("reference_time")
    @classmethod
    def validate_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure reference_time is timezone-aware."""
        if v.tzinfo is None:
            raise ValueError("reference_time must be timezone-aware")
        return v

    def _get_graphiti_source_type(self) -> GraphitiSourceType:
        """Map episode source to appropriate Graphiti source type.

        Returns:
            The corresponding GraphitiSourceType for this episode's source.
        """
        # Slack messages map to MESSAGE type
        if self.source == EpisodeSource.SLACK:
            return GraphitiSourceType.MESSAGE
        # Notion and other document-based sources map to TEXT type
        elif self.source in (EpisodeSource.NOTION, EpisodeSource.GITHUB):
            return GraphitiSourceType.TEXT
        # Jira issues typically have structured data
        elif self.source == EpisodeSource.JIRA:
            return GraphitiSourceType.JSON
        # Default to TEXT for manual and unknown sources
        return GraphitiSourceType.TEXT

    def _get_source_description(self) -> str:
        """Generate a descriptive source string for Graphiti.

        Returns:
            A human-readable description of the episode source.
        """
        source_descriptions = {
            EpisodeSource.SLACK: "Slack message",
            EpisodeSource.NOTION: "Notion page",
            EpisodeSource.GITHUB: "GitHub content",
            EpisodeSource.JIRA: "Jira issue",
            EpisodeSource.MANUAL: "Manual entry",
        }
        base_description = source_descriptions.get(self.source, "Unknown source")

        # Add channel/context info from metadata if available
        if self.source == EpisodeSource.SLACK:
            channel_name = self.metadata.get("channel_name")
            if channel_name:
                return f"{base_description} from #{channel_name}"

        return base_description

    def to_graphiti_params(self) -> dict[str, Any]:
        """Convert episode to parameters for graphiti-core add_episode().

        This method transforms the standardized Episode format into the
        parameter dictionary expected by Graphiti's add_episode() method.

        Returns:
            A dictionary containing:
                - name: Episode name/title
                - episode_body: The content body
                - source: Graphiti EpisodeType value (text, message, json)
                - source_description: Human-readable source description
                - reference_time: Timezone-aware datetime
                - group_id: Logical grouping identifier

        Example:
            >>> episode = Episode(
            ...     name="Message from @alice",
            ...     body="Hello world",
            ...     source=EpisodeSource.SLACK,
            ...     reference_time=datetime.now(timezone.utc),
            ...     group_id="C01234567",
            ... )
            >>> params = episode.to_graphiti_params()
            >>> await graphiti_client.add_episode(**params)
        """
        return {
            "name": self.name,
            "episode_body": self.body,
            "source": self._get_graphiti_source_type().value,
            "source_description": self._get_source_description(),
            "reference_time": self.reference_time,
            "group_id": self.group_id,
        }

    def to_raw_episode_params(self) -> dict[str, Any]:
        """Convert episode to parameters for RawEpisode (bulk ingestion).

        This method transforms the Episode into parameters suitable for
        creating a graphiti_core.utils.bulk_utils.RawEpisode object for
        bulk ingestion via add_episode_bulk().

        Returns:
            A dictionary containing:
                - name: Episode name/title
                - content: The content body (note: RawEpisode uses 'content')
                - source: Graphiti EpisodeType value
                - source_description: Human-readable source description
                - reference_time: Timezone-aware datetime

        Note:
            group_id is passed separately to add_episode_bulk(), not per episode.
        """
        return {
            "name": self.name,
            "content": self.body,
            "source": self._get_graphiti_source_type().value,
            "source_description": self._get_source_description(),
            "reference_time": self.reference_time,
        }

    @classmethod
    def create_slack_episode(
        cls,
        name: str,
        body: str,
        reference_time: datetime,
        channel_id: str,
        message_ts: str,
        user_id: str | None = None,
        thread_ts: str | None = None,
        channel_name: str | None = None,
        **extra_metadata: Any,
    ) -> "Episode":
        """Factory method to create a Slack-specific episode.

        This is a convenience method that properly structures Slack-specific
        metadata and ensures consistent episode creation from Slack data.

        Args:
            name: Episode name/title.
            body: Message content.
            reference_time: When the message was posted.
            channel_id: Slack channel ID (used as group_id).
            message_ts: Slack message timestamp.
            user_id: Slack user ID who posted the message.
            thread_ts: Thread timestamp if this is a threaded reply.
            channel_name: Human-readable channel name.
            **extra_metadata: Additional metadata to include.

        Returns:
            A properly configured Episode instance for Slack data.
        """
        metadata = {
            "message_ts": message_ts,
            "user_id": user_id,
            "thread_ts": thread_ts,
            "channel_name": channel_name,
            **extra_metadata,
        }
        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return cls(
            name=name,
            body=body,
            source=EpisodeSource.SLACK,
            reference_time=reference_time,
            group_id=channel_id,
            metadata=metadata,
        )
