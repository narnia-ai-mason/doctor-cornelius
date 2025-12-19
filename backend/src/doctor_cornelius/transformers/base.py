"""Base transformer interface for converting raw collected data into Episode format.

This module defines the abstract base class for all transformers in the Doctor Cornelius
system. Transformers are responsible for converting raw data items collected from various
sources (Slack, Notion, GitHub, etc.) into a standardized Episode format that can be
ingested into the knowledge graph.

Example usage:
    class SlackTransformer(BaseTransformer[SlackRawDataItem]):
        async def transform(self, item: SlackRawDataItem) -> Episode | None:
            if self.should_skip(item):
                return None
            return Episode(
                name=f"slack-message-{item.message_id}",
                body=item.text,
                source="slack",
                reference_time=item.timestamp,
                ...
            )
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from doctor_cornelius.schemas.episode import Episode

# Type variable for raw data items from collectors
# Each collector produces its own RawDataItem type (e.g., SlackRawDataItem, NotionRawDataItem)
RawDataItemT = TypeVar("RawDataItemT")


class BaseTransformer(ABC, Generic[RawDataItemT]):
    """Abstract base class for transforming raw collected data into Episode format.

    Transformers take raw data items from collectors and convert them into standardized
    Episode objects that can be ingested into the Graphiti knowledge graph. Each data
    source (Slack, Notion, GitHub, etc.) should have its own transformer implementation.

    Type Parameters:
        RawDataItemT: The type of raw data item this transformer handles.
                      Should match the output type of the corresponding collector.

    Attributes:
        source_name: Identifier for the data source (e.g., "slack", "notion").

    Example:
        >>> class SlackTransformer(BaseTransformer[SlackRawDataItem]):
        ...     source_name = "slack"
        ...
        ...     async def transform(self, item: SlackRawDataItem) -> Episode | None:
        ...         if self.should_skip(item):
        ...             return None
        ...         return Episode(...)
    """

    source_name: str = "unknown"

    @abstractmethod
    async def transform(self, item: RawDataItemT) -> Episode | None:
        """Transform a single raw data item into an Episode.

        This method should handle all the logic for converting source-specific
        data into the standardized Episode format, including:
        - Extracting relevant fields (name, body, metadata)
        - Resolving references (e.g., user mentions, channel names)
        - Setting appropriate timestamps
        - Generating unique identifiers

        Args:
            item: The raw data item to transform. The type depends on the
                  specific transformer implementation.

        Returns:
            An Episode object if the transformation is successful, or None if
            the item should be skipped (e.g., system messages, bot messages).

        Raises:
            TransformError: If the transformation fails due to invalid data.

        Example:
            >>> episode = await transformer.transform(slack_message)
            >>> if episode:
            ...     await graph_client.ingest_episode(episode)
        """
        ...

    async def transform_batch(
        self,
        items: list[RawDataItemT],
    ) -> list[Episode]:
        """Transform multiple raw data items into Episodes.

        This method processes a batch of items and returns only the successfully
        transformed Episodes (items that should be skipped return None and are
        filtered out).

        The default implementation processes items sequentially. Subclasses may
        override this to implement concurrent processing for better performance,
        while respecting rate limits and resource constraints.

        Args:
            items: A list of raw data items to transform.

        Returns:
            A list of Episode objects. Items that were skipped or failed
            transformation are not included in the result.

        Example:
            >>> messages = await slack_collector.collect(channel_id, since)
            >>> episodes = await transformer.transform_batch(list(messages))
            >>> await graph_client.ingest_episodes_batch(episodes)
        """
        episodes: list[Episode] = []
        for item in items:
            episode = await self.transform(item)
            if episode is not None:
                episodes.append(episode)
        return episodes

    @abstractmethod
    def should_skip(self, item: RawDataItemT) -> bool:
        """Determine if a raw data item should be skipped during transformation.

        This method is used to filter out items that should not be converted
        into Episodes. Common reasons to skip items include:
        - System-generated messages (channel_join, channel_leave, etc.)
        - Bot-generated messages
        - Empty or invalid content
        - Duplicate items

        Args:
            item: The raw data item to evaluate.

        Returns:
            True if the item should be skipped, False if it should be processed.

        Example:
            >>> # In SlackTransformer
            >>> def should_skip(self, item: SlackRawDataItem) -> bool:
            ...     # Skip system messages
            ...     if item.subtype in ("channel_join", "channel_leave"):
            ...         return True
            ...     # Skip bot messages
            ...     if item.bot_id is not None:
            ...         return True
            ...     return False
        """
        ...

    def __repr__(self) -> str:
        """Return a string representation of the transformer."""
        return f"{self.__class__.__name__}(source={self.source_name!r})"
