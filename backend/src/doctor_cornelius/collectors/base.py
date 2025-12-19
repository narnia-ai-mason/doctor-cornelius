"""Base collector interface for Doctor Cornelius data collection.

This module defines the abstract base class and schemas for all data collectors.
Collectors are responsible for fetching raw data from various sources (Slack, Notion,
GitHub, etc.) and yielding standardized RawDataItem objects for transformation.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Enumeration of supported data source types."""

    SLACK = "slack"
    NOTION = "notion"
    GITHUB = "github"
    JIRA = "jira"


class RawDataItem(BaseModel):
    """Schema for raw data collected from any source.

    This is the standardized output format for all collectors. Each RawDataItem
    represents a single piece of content (message, document, issue, etc.) that
    will be transformed into an Episode for the knowledge graph.

    Attributes:
        source_type: The type of source this data came from.
        source_id: Unique identifier within the source (e.g., message ts, page id).
        source_name: Human-readable name of the source (e.g., channel name).
        group_id: Identifier for grouping related items (e.g., channel_id, repo name).
        content: The main textual content of the item.
        author_id: Identifier of the content author.
        author_name: Display name of the content author.
        timestamp: When this content was created/posted.
        parent_id: For threaded/nested content, the parent item's source_id.
        thread_ts: Thread timestamp for threaded conversations (Slack-specific).
        reply_count: Number of replies if this is a parent message.
        metadata: Additional source-specific metadata.
        raw_data: The original unprocessed data from the source API.
    """

    source_type: SourceType = Field(description="The type of data source")
    source_id: str = Field(description="Unique identifier within the source")
    source_name: str = Field(description="Human-readable source name")
    group_id: str = Field(description="Grouping identifier (e.g., channel_id)")
    content: str = Field(description="Main textual content")
    author_id: str = Field(description="Author's unique identifier")
    author_name: str = Field(default="", description="Author's display name")
    timestamp: datetime = Field(description="When the content was created")
    parent_id: str | None = Field(default=None, description="Parent item ID for nested content")
    thread_ts: str | None = Field(
        default=None, description="Thread timestamp for threaded conversations"
    )
    reply_count: int = Field(default=0, description="Number of replies")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Original API response data")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class DataSource(BaseModel):
    """Schema representing an available data source.

    This represents a collectible source like a Slack channel, Notion database,
    GitHub repository, or Jira project.

    Attributes:
        source_type: The type of data source.
        source_id: Unique identifier for this source.
        name: Human-readable name of the source.
        description: Optional description or topic.
        is_accessible: Whether the collector has access to this source.
        member_count: Number of members (if applicable).
        metadata: Additional source-specific information.
    """

    source_type: SourceType = Field(description="The type of data source")
    source_id: str = Field(description="Unique identifier for this source")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="Description or topic")
    is_accessible: bool = Field(
        default=True, description="Whether the collector can access this source"
    )
    member_count: int | None = Field(default=None, description="Number of members if applicable")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional source information"
    )


class CollectionConfig(BaseModel):
    """Configuration for a data collection operation.

    This schema defines the parameters for collecting data from a source,
    including time ranges, filtering options, and pagination settings.

    Attributes:
        source_ids: Specific source IDs to collect from. If empty, collect from all.
        start_time: Collect items created after this time (inclusive).
        end_time: Collect items created before this time (exclusive).
        include_threads: Whether to include threaded/nested content.
        include_replies: Whether to include reply messages.
        batch_size: Number of items to fetch per API request.
        max_items: Maximum total items to collect (None for unlimited).
        rate_limit_delay: Delay in seconds between API requests.
    """

    source_ids: list[str] = Field(
        default_factory=list,
        description="Specific source IDs to collect from. Empty means all.",
    )
    start_time: datetime | None = Field(default=None, description="Collect items after this time")
    end_time: datetime | None = Field(default=None, description="Collect items before this time")
    include_threads: bool = Field(default=True, description="Include threaded/nested content")
    include_replies: bool = Field(default=True, description="Include reply messages")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Items per API request")
    max_items: int | None = Field(default=None, ge=1, description="Maximum items to collect")
    rate_limit_delay: float = Field(
        default=0.5, ge=0.0, description="Delay between API requests in seconds"
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


@dataclass
class CollectionStats:
    """Statistics for a collection operation.

    Tracks the progress and results of a data collection run.

    Attributes:
        sources_processed: Number of sources that were processed.
        items_collected: Total number of items collected.
        items_skipped: Number of items skipped (e.g., system messages).
        errors: List of error messages encountered during collection.
        start_time: When the collection started.
        end_time: When the collection completed.
    """

    sources_processed: int = 0
    items_collected: int = 0
    items_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate the duration of the collection in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        total = self.items_collected + self.items_skipped + len(self.errors)
        if total == 0:
            return 100.0
        return (self.items_collected / total) * 100.0


# Type variable for source-specific configuration
ConfigT = TypeVar("ConfigT", bound=CollectionConfig)


class BaseCollector(ABC, Generic[ConfigT]):
    """Abstract base class for all data collectors.

    Collectors are responsible for connecting to external data sources and
    fetching raw data items. Each collector implementation handles the specifics
    of its source's API, authentication, and data format.

    This class uses generics to allow subclasses to specify their own
    configuration type that extends CollectionConfig.

    Type Parameters:
        ConfigT: The configuration type for this collector, must extend CollectionConfig.

    Example:
        ```python
        class SlackCollector(BaseCollector[SlackCollectionConfig]):
            async def list_sources(self) -> list[DataSource]:
                # List available Slack channels
                ...

            async def collect(
                self, 
                config: SlackCollectionConfig
            ) -> AsyncGenerator[RawDataItem, None]:
                # Yield messages from Slack channels
                ...
        ```
    """

    def __init__(self, source_type: SourceType) -> None:
        """Initialize the base collector.

        Args:
            source_type: The type of source this collector handles.
        """
        self._source_type = source_type
        self._stats = CollectionStats()

    @property
    def source_type(self) -> SourceType:
        """Get the source type for this collector."""
        return self._source_type

    @property
    def stats(self) -> CollectionStats:
        """Get the current collection statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset the collection statistics for a new run."""
        self._stats = CollectionStats()

    @abstractmethod
    async def list_sources(self) -> list[DataSource]:
        """List all available data sources that can be collected.

        This method should return all sources (e.g., channels, databases, repos)
        that the collector has access to, applying any security filters as needed.

        Returns:
            A list of DataSource objects representing available sources.

        Raises:
            ConnectionError: If unable to connect to the source API.
            PermissionError: If authentication fails or lacks required permissions.
        """
        ...

    @abstractmethod
    async def collect(self, config: ConfigT) -> AsyncGenerator[RawDataItem, None]:
        """Collect data items from the configured sources.

        This is an async generator that yields RawDataItem objects as they are
        fetched from the source. It handles pagination, rate limiting, and
        error recovery internally.

        Args:
            config: Configuration for this collection run.

        Yields:
            RawDataItem objects representing collected content.

        Raises:
            ConnectionError: If unable to connect to the source API.
            PermissionError: If authentication fails or lacks required permissions.
            ValueError: If the configuration is invalid.

        Example:
            ```python
            config = CollectionConfig(
                start_time=datetime(2024, 1, 1),
                include_threads=True
            )
            async for item in collector.collect(config):
                process_item(item)
            ```
        """
        # This is required to make this an async generator
        # Subclasses will implement the actual logic
        if False:  # pragma: no cover
            yield  # type: ignore[misc]

    async def validate_connection(self) -> bool:
        """Validate that the collector can connect to its data source.

        This method should perform a lightweight check to verify connectivity
        and authentication without fetching significant data.

        Returns:
            True if the connection is valid, False otherwise.
        """
        try:
            # Default implementation tries to list sources
            await self.list_sources()
            return True
        except Exception:
            return False

    async def collect_all(self, config: ConfigT) -> tuple[list[RawDataItem], CollectionStats]:
        """Collect all items and return them as a list with statistics.

        This is a convenience method that collects all items into memory.
        For large datasets, prefer using the collect() generator directly.

        Args:
            config: Configuration for this collection run.

        Returns:
            A tuple of (list of collected items, collection statistics).

        Warning:
            This method loads all items into memory. For large datasets,
            use the collect() async generator instead.
        """
        self.reset_stats()
        self._stats.start_time = datetime.now()

        items: list[RawDataItem] = []
        async for item in self.collect(config):
            items.append(item)
            self._stats.items_collected += 1

        self._stats.end_time = datetime.now()
        return items, self._stats


# Export public API
__all__ = [
    "BaseCollector",
    "CollectionConfig",
    "CollectionStats",
    "DataSource",
    "RawDataItem",
    "SourceType",
    "ConfigT",
]
