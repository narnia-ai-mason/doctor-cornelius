"""Slack transformer for converting raw Slack data into Episode format.

This module implements the SlackTransformer class that converts raw Slack messages
collected by SlackCollector into standardized Episode objects for knowledge graph
ingestion.

Key features:
- Skips system messages (channel_join, channel_leave, bot_add, etc.)
- Resolves user mentions (<@U12345>) to readable names
- Groups thread messages as conversation episodes
- Preserves Slack-specific metadata for traceability

Example usage:
    transformer = SlackTransformer(user_resolver=async_user_lookup)
    episodes = await transformer.transform_batch(slack_messages)
    for episode in episodes:
        await graph_client.ingest_episode(episode)
"""

import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from doctor_cornelius.collectors.base import RawDataItem
from doctor_cornelius.schemas.episode import Episode
from doctor_cornelius.transformers.base import BaseTransformer

# System message subtypes that should be skipped during transformation
SYSTEM_MESSAGE_SUBTYPES: frozenset[str] = frozenset(
    {
        "channel_join",
        "channel_leave",
        "channel_topic",
        "channel_purpose",
        "channel_name",
        "channel_archive",
        "channel_unarchive",
        "group_join",
        "group_leave",
        "group_topic",
        "group_purpose",
        "group_name",
        "group_archive",
        "group_unarchive",
        "bot_add",
        "bot_remove",
        "bot_enable",
        "bot_disable",
        "ekm_access_denied",
        "file_share",
        "file_comment",
        "file_mention",
        "pinned_item",
        "unpinned_item",
        "tombstone",
        "thread_broadcast",
    }
)

# Regex pattern to match Slack user mentions: <@U12345678> or <@U12345678|username>
USER_MENTION_PATTERN = re.compile(r"<@([A-Z0-9]+)(?:\|([^>]+))?>")

# Regex pattern to match Slack channel mentions: <#C12345678> or <#C12345678|channel-name>
CHANNEL_MENTION_PATTERN = re.compile(r"<#([A-Z0-9]+)(?:\|([^>]+))?>")

# Regex pattern to match Slack links: <http://example.com|text>
LINK_PATTERN = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")

# Maximum length for episode names
MAX_EPISODE_NAME_LENGTH = 100


# Type alias for user resolver function
UserResolver = Callable[[str], Awaitable[str | None]]


class SlackTransformer(BaseTransformer[RawDataItem]):
    """Transformer for converting Slack messages into Episode format.

    This transformer handles the conversion of raw Slack message data into
    standardized Episode objects suitable for ingestion into the Graphiti
    knowledge graph. It includes logic for:

    - Filtering out system messages that don't contain meaningful content
    - Resolving user mentions to readable names
    - Creating meaningful episode names from message content
    - Preserving thread context and metadata

    Attributes:
        source_name: Identifier for the data source ("slack").
        user_resolver: Optional async function to resolve user IDs to names.
        user_cache: Internal cache for resolved user names.

    Example:
        >>> async def lookup_user(user_id: str) -> str | None:
        ...     return await slack_client.users_info(user=user_id)
        ...
        >>> transformer = SlackTransformer(user_resolver=lookup_user)
        >>> episode = await transformer.transform(slack_message)
    """

    source_name: str = "slack"

    def __init__(
        self,
        user_resolver: UserResolver | None = None,
        channel_resolver: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        """Initialize the Slack transformer.

        Args:
            user_resolver: Optional async function that takes a user ID and returns
                the user's display name. If not provided, user mentions will use
                the fallback name from the mention syntax or the raw user ID.
            channel_resolver: Optional async function that takes a channel ID and
                returns the channel name. If not provided, channel mentions will
                use the fallback name from the mention syntax.
        """
        self._user_resolver = user_resolver
        self._channel_resolver = channel_resolver
        self._user_cache: dict[str, str] = {}
        self._channel_cache: dict[str, str] = {}

    async def transform(self, item: RawDataItem) -> Episode | None:
        """Transform a single Slack message into an Episode.

        This method handles the full transformation of a raw Slack message,
        including:
        1. Checking if the message should be skipped (system messages, etc.)
        2. Resolving user mentions to readable names
        3. Creating a meaningful episode name
        4. Building the Episode object with proper metadata

        Args:
            item: The raw Slack message data item to transform.

        Returns:
            An Episode object if transformation is successful, or None if
            the message should be skipped.

        Raises:
            ValueError: If the item has invalid or missing required fields.
        """
        # Skip system messages and other non-content items
        if self.should_skip(item):
            return None

        # Resolve user mentions in the message content
        resolved_content = await self._resolve_mentions(item.content)

        # Create a meaningful episode name
        episode_name = self._create_episode_name(item, resolved_content)

        # Build metadata dictionary
        metadata = self._build_metadata(item)

        # Ensure timestamp is timezone-aware
        reference_time = self._ensure_timezone_aware(item.timestamp)

        # Create and return the Episode using the factory method
        return Episode.create_slack_episode(
            name=episode_name,
            body=resolved_content,
            reference_time=reference_time,
            channel_id=item.group_id,
            message_ts=item.source_id,
            user_id=item.author_id,
            user_name=item.author_name or None,
            thread_ts=item.thread_ts,
            channel_name=metadata.get("channel_name"),
            reply_count=item.reply_count,
        )

    def should_skip(self, item: RawDataItem) -> bool:
        """Determine if a Slack message should be skipped during transformation.

        Messages are skipped if they are:
        - System messages (channel_join, channel_leave, bot_add, etc.)
        - Empty or whitespace-only content
        - From bots (optional, configurable)

        Args:
            item: The raw Slack message data item to evaluate.

        Returns:
            True if the message should be skipped, False otherwise.
        """
        # Check for system message subtypes
        subtype = item.metadata.get("subtype")
        if subtype and subtype in SYSTEM_MESSAGE_SUBTYPES:
            return True

        # Check for empty content
        if not item.content or not item.content.strip():
            return True

        # Skip messages from bots (check both bot_id and is_bot flag)
        if item.metadata.get("bot_id") is not None:
            return True

        return item.raw_data.get("bot_id") is not None

    async def _resolve_mentions(self, content: str) -> str:
        """Resolve user and channel mentions in message content.

        Converts Slack-specific mention formats to human-readable text:
        - <@U12345678> -> @username
        - <@U12345678|display_name> -> @display_name
        - <#C12345678|channel-name> -> #channel-name
        - <https://example.com|text> -> text (https://example.com)

        Args:
            content: The raw message content with Slack mention syntax.

        Returns:
            The content with mentions resolved to readable names.
        """
        # Resolve user mentions
        content = await self._resolve_user_mentions(content)

        # Resolve channel mentions
        content = await self._resolve_channel_mentions(content)

        # Resolve links to readable format
        content = self._resolve_links(content)

        return content

    async def _resolve_user_mentions(self, content: str) -> str:
        """Resolve user mentions (<@U12345678>) to @username format.

        Args:
            content: The message content containing user mentions.

        Returns:
            Content with user mentions resolved to readable names.
        """
        matches = USER_MENTION_PATTERN.findall(content)

        for user_id, fallback_name in matches:
            # Try to get the resolved name
            resolved_name = await self._get_user_name(user_id, fallback_name)
            # Replace the mention with the resolved name
            pattern = f"<@{user_id}(?:\\|[^>]+)?>"
            content = re.sub(pattern, f"@{resolved_name}", content)

        return content

    async def _get_user_name(self, user_id: str, fallback: str | None = None) -> str:
        """Get the display name for a user ID, with caching.

        Args:
            user_id: The Slack user ID to look up.
            fallback: Optional fallback name from the mention syntax.

        Returns:
            The user's display name, or fallback if not resolvable.
        """
        # Check cache first
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        # Try to resolve via the user resolver
        if self._user_resolver:
            try:
                resolved_name = await self._user_resolver(user_id)
                if resolved_name:
                    self._user_cache[user_id] = resolved_name
                    return resolved_name
            except Exception:
                # Silently fail and use fallback
                pass

        # Use fallback or user ID
        result = fallback if fallback else user_id
        self._user_cache[user_id] = result
        return result

    async def _resolve_channel_mentions(self, content: str) -> str:
        """Resolve channel mentions (<#C12345678>) to #channel-name format.

        Args:
            content: The message content containing channel mentions.

        Returns:
            Content with channel mentions resolved to readable names.
        """
        matches = CHANNEL_MENTION_PATTERN.findall(content)

        for channel_id, fallback_name in matches:
            resolved_name = await self._get_channel_name(channel_id, fallback_name)
            pattern = f"<#{channel_id}(?:\\|[^>]+)?>"
            content = re.sub(pattern, f"#{resolved_name}", content)

        return content

    async def _get_channel_name(self, channel_id: str, fallback: str | None = None) -> str:
        """Get the name for a channel ID, with caching.

        Args:
            channel_id: The Slack channel ID to look up.
            fallback: Optional fallback name from the mention syntax.

        Returns:
            The channel name, or fallback if not resolvable.
        """
        # Check cache first
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]

        # Try to resolve via the channel resolver
        if self._channel_resolver:
            try:
                resolved_name = await self._channel_resolver(channel_id)
                if resolved_name:
                    self._channel_cache[channel_id] = resolved_name
                    return resolved_name
            except Exception:
                # Silently fail and use fallback
                pass

        # Use fallback or channel ID
        result = fallback if fallback else channel_id
        self._channel_cache[channel_id] = result
        return result

    def _resolve_links(self, content: str) -> str:
        """Resolve Slack link syntax to readable format.

        Converts <https://example.com|text> to text (https://example.com)
        and <https://example.com> to https://example.com

        Args:
            content: The message content containing Slack link syntax.

        Returns:
            Content with links in readable format.
        """

        def replace_link(match: re.Match[str]) -> str:
            url = match.group(1)
            text = match.group(2)
            if text:
                return f"{text} ({url})"
            return url

        return LINK_PATTERN.sub(replace_link, content)

    def _create_episode_name(self, item: RawDataItem, resolved_content: str) -> str:
        """Create a meaningful episode name from the message.

        The episode name format depends on the message context:
        - Thread reply: "Reply from @user in #channel (thread)"
        - Regular message: "Message from @user in #channel"

        The name includes a preview of the content if it fits within limits.

        Args:
            item: The raw Slack message data item.
            resolved_content: The content with mentions already resolved.

        Returns:
            A descriptive episode name suitable for display.
        """
        # Determine user display name
        author_display = item.author_name if item.author_name else item.author_id

        # Determine if this is a thread reply
        is_thread_reply = item.thread_ts is not None and item.thread_ts != item.source_id

        # Get channel name from metadata or source_name
        channel_name = item.metadata.get("channel_name", item.source_name)

        # Build the base name
        if is_thread_reply:
            base_name = f"Reply from @{author_display} in #{channel_name}"
        else:
            base_name = f"Message from @{author_display} in #{channel_name}"

        # Add thread indicator if it's a thread parent with replies
        if item.reply_count > 0:
            base_name = f"{base_name} ({item.reply_count} replies)"

        # Add content preview if there's room
        preview = self._create_content_preview(resolved_content)
        if preview:
            full_name = f"{base_name}: {preview}"
            if len(full_name) <= MAX_EPISODE_NAME_LENGTH:
                return full_name

        return base_name[:MAX_EPISODE_NAME_LENGTH]

    def _create_content_preview(self, content: str, max_length: int = 50) -> str | None:
        """Create a short preview of the message content.

        Args:
            content: The full message content.
            max_length: Maximum length of the preview.

        Returns:
            A truncated preview of the content, or None if content is too short.
        """
        # Clean up the content - remove extra whitespace
        cleaned = " ".join(content.split())

        if len(cleaned) <= 10:  # Too short to be meaningful
            return None

        if len(cleaned) <= max_length:
            return cleaned

        # Truncate and add ellipsis
        return cleaned[: max_length - 3] + "..."

    def _build_metadata(self, item: RawDataItem) -> dict[str, Any]:
        """Build the metadata dictionary for the Episode.

        This preserves Slack-specific metadata for traceability and
        potential future use (e.g., linking back to original messages).

        Args:
            item: The raw Slack message data item.

        Returns:
            A dictionary of metadata to include in the Episode.
        """
        metadata: dict[str, Any] = {
            "channel_id": item.group_id,
            "channel_name": item.metadata.get("channel_name", item.source_name),
            "user_id": item.author_id,
            "message_ts": item.source_id,
        }

        # Add optional fields if present
        if item.author_name:
            metadata["user_name"] = item.author_name

        if item.thread_ts:
            metadata["thread_ts"] = item.thread_ts

        if item.reply_count > 0:
            metadata["reply_count"] = item.reply_count

        if item.parent_id:
            metadata["parent_id"] = item.parent_id

        # Include any reactions or attachments info from raw_data
        if item.raw_data.get("reactions"):
            metadata["reactions"] = item.raw_data["reactions"]

        if item.raw_data.get("attachments"):
            metadata["has_attachments"] = True

        if item.raw_data.get("files"):
            metadata["has_files"] = True

        return metadata

    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Ensure the datetime is timezone-aware.

        Args:
            dt: The datetime to check/convert.

        Returns:
            A timezone-aware datetime (defaults to UTC if naive).
        """
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    async def transform_thread(self, thread_messages: list[RawDataItem]) -> Episode | None:
        """Transform a thread of messages into a single conversation Episode.

        This method aggregates a thread (parent message + replies) into a
        single Episode that represents the full conversation context.

        Args:
            thread_messages: List of messages in the thread, ordered by timestamp.
                The first message should be the thread parent.

        Returns:
            A single Episode representing the entire thread conversation,
            or None if no valid messages in the thread.
        """
        if not thread_messages:
            return None

        # Filter out messages that should be skipped
        valid_messages = [msg for msg in thread_messages if not self.should_skip(msg)]
        if not valid_messages:
            return None

        # Use the parent message (first one) for base metadata
        parent = valid_messages[0]

        # Build the conversation body
        conversation_parts: list[str] = []
        for msg in valid_messages:
            resolved_content = await self._resolve_mentions(msg.content)
            author_name = msg.author_name or msg.author_id
            conversation_parts.append(f"@{author_name}: {resolved_content}")

        conversation_body = "\n\n".join(conversation_parts)

        # Create episode name for the thread
        channel_name = parent.metadata.get("channel_name", parent.source_name)
        author_display = parent.author_name or parent.author_id
        episode_name = (
            f"Thread conversation in #{channel_name} "
            f"started by @{author_display} ({len(valid_messages)} messages)"
        )
        episode_name = episode_name[:MAX_EPISODE_NAME_LENGTH]

        # Use parent's timestamp and thread_ts
        reference_time = self._ensure_timezone_aware(parent.timestamp)

        return Episode.create_slack_episode(
            name=episode_name,
            body=conversation_body,
            reference_time=reference_time,
            channel_id=parent.group_id,
            message_ts=parent.source_id,
            user_id=parent.author_id,
            user_name=parent.author_name or None,
            thread_ts=parent.thread_ts or parent.source_id,
            channel_name=channel_name,
            reply_count=len(valid_messages) - 1,
            is_thread_conversation=True,
            participant_count=len({msg.author_id for msg in valid_messages}),
        )

    def clear_caches(self) -> None:
        """Clear the user and channel name caches.

        This is useful when you want to refresh cached names, for example
        when a user's display name has changed.
        """
        self._user_cache.clear()
        self._channel_cache.clear()


# Export public API
__all__ = [
    "SlackTransformer",
    "SYSTEM_MESSAGE_SUBTYPES",
    "UserResolver",
]
