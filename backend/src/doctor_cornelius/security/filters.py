"""Security filters for data collection.

This module provides filtering mechanisms to ensure that only appropriate
data sources are collected from. The primary use case is Slack channel
filtering to exclude archived, external, and potentially sensitive channels.
"""

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class SlackChannelFilter:
    """Filter for determining which Slack channels should be included in data collection.

    This filter implements security best practices by excluding:
    - Archived channels (no longer active)
    - External shared channels (is_ext_shared) - channels shared with external organizations
    - Shared channels (is_shared) - channels shared across workspaces
    - Channels with blocked name prefixes (configurable)

    Attributes:
        blocked_prefixes: List of channel name prefixes to exclude.
            Defaults to ["external-", "guest-"].
        allow_archived: If True, includes archived channels. Defaults to False.
        allow_external_shared: If True, includes externally shared channels.
            Defaults to False.
        allow_shared: If True, includes shared channels. Defaults to False.

    Example:
        >>> filter = SlackChannelFilter()
        >>> channel = {"name": "engineering", "is_archived": False, "is_ext_shared": False}
        >>> filter.should_include(channel)
        True

        >>> external_channel = {
                "name": "external-partner", 
                "is_archived": False, 
                "is_ext_shared": True
            }
        >>> filter.should_include(external_channel)
        False
    """

    blocked_prefixes: list[str] = field(
        default_factory=lambda: ["external-", "guest-"]
    )
    allow_archived: bool = False
    allow_external_shared: bool = False
    allow_shared: bool = False

    def should_include(self, channel: dict[str, Any]) -> bool:
        """Determine if a channel should be included in data collection.

        Args:
            channel: A dictionary containing channel information from the Slack API.
                Expected keys include:
                - name (str): The channel name
                - is_archived (bool): Whether the channel is archived
                - is_ext_shared (bool): Whether the channel is externally shared
                - is_shared (bool): Whether the channel is shared across workspaces

        Returns:
            True if the channel passes all filter criteria and should be included,
            False otherwise.

        Note:
            Missing keys in the channel dictionary are treated as False for
            boolean fields and empty string for the name field.
        """
        channel_name = channel.get("name", "")
        channel_id = channel.get("id", "unknown")

        # Check if channel is archived
        if not self.allow_archived and channel.get("is_archived", False):
            logger.debug(
                "channel_excluded_archived",
                channel_id=channel_id,
                channel_name=channel_name,
            )
            return False

        # Check if channel is externally shared
        if not self.allow_external_shared and channel.get("is_ext_shared", False):
            logger.debug(
                "channel_excluded_external_shared",
                channel_id=channel_id,
                channel_name=channel_name,
            )
            return False

        # Check if channel is shared across workspaces
        if not self.allow_shared and channel.get("is_shared", False):
            logger.debug(
                "channel_excluded_shared",
                channel_id=channel_id,
                channel_name=channel_name,
            )
            return False

        # Check if channel name starts with a blocked prefix
        if self._has_blocked_prefix(channel_name):
            logger.debug(
                "channel_excluded_blocked_prefix",
                channel_id=channel_id,
                channel_name=channel_name,
                blocked_prefixes=self.blocked_prefixes,
            )
            return False

        return True

    def _has_blocked_prefix(self, channel_name: str) -> bool:
        """Check if the channel name starts with any blocked prefix.

        Args:
            channel_name: The name of the channel to check.

        Returns:
            True if the channel name starts with a blocked prefix, False otherwise.
        """
        return any(
            channel_name.startswith(prefix) for prefix in self.blocked_prefixes
        )

    def filter_channels(self, channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter a list of channels, returning only those that should be included.

        Args:
            channels: A list of channel dictionaries from the Slack API.

        Returns:
            A filtered list containing only channels that pass all filter criteria.
        """
        included = [ch for ch in channels if self.should_include(ch)]
        excluded_count = len(channels) - len(included)

        if excluded_count > 0:
            logger.info(
                "channels_filtered",
                total_channels=len(channels),
                included_channels=len(included),
                excluded_channels=excluded_count,
            )

        return included

    def get_exclusion_reason(self, channel: dict[str, Any]) -> str | None:
        """Get a human-readable reason why a channel would be excluded.

        Args:
            channel: A dictionary containing channel information from the Slack API.

        Returns:
            A string describing why the channel would be excluded, or None if
            the channel would be included.
        """
        channel_name = channel.get("name", "")

        if not self.allow_archived and channel.get("is_archived", False):
            return "Channel is archived"

        if not self.allow_external_shared and channel.get("is_ext_shared", False):
            return "Channel is externally shared with other organizations"

        if not self.allow_shared and channel.get("is_shared", False):
            return "Channel is shared across workspaces"

        if self._has_blocked_prefix(channel_name):
            matching_prefix = next(
                (p for p in self.blocked_prefixes if channel_name.startswith(p)),
                None,
            )
            return f"Channel name starts with blocked prefix: '{matching_prefix}'"

        return None
