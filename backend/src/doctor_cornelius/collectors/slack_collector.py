"""Slack data collector for Doctor Cornelius.

This module implements the SlackCollector class for fetching messages from Slack
channels. It supports pagination, thread reply collection, and rate limiting.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import structlog
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from doctor_cornelius.collectors.base import (
    BaseCollector,
    CollectionConfig,
    DataSource,
    RawDataItem,
    SourceType,
)
from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.security.filters import SlackChannelFilter

logger = structlog.get_logger()


class SlackCollector(BaseCollector[CollectionConfig]):
    """Collector for fetching messages from Slack channels.

    This collector uses the Slack Web API to fetch channel messages and thread
    replies. It implements pagination, rate limiting, and security filtering
    to ensure safe and efficient data collection.

    Features:
        - List channels the bot is a member of with security filtering
        - Collect messages with configurable time ranges
        - Fetch thread replies for threaded conversations
        - Rate limiting with configurable delay between requests
        - Retry logic for transient API errors

    Example:
        ```python
        collector = SlackCollector()
        sources = await collector.list_sources()

        config = CollectionConfig(
            source_ids=[s.source_id for s in sources[:3]],
            include_threads=True,
            start_time=datetime(2024, 1, 1)
        )

        async for item in collector.collect(config):
            print(f"Message from {item.author_name}: {item.content[:50]}")
        ```
    """

    def __init__(
        self,
        settings: Settings | None = None,
        channel_filter: SlackChannelFilter | None = None,
    ) -> None:
        """Initialize the Slack collector.

        Args:
            settings: Application settings. If None, loads from environment.
            channel_filter: Filter for determining which channels to include.
                If None, creates a default filter with blocked prefixes from settings.
        """
        super().__init__(SourceType.SLACK)

        self._settings = settings or get_settings()
        self._client = AsyncWebClient(token=self._settings.slack.bot_token.get_secret_value())

        # Initialize channel filter with blocked prefixes from settings
        if channel_filter is None:
            self._channel_filter = SlackChannelFilter(
                blocked_prefixes=self._settings.collector.blocked_channel_prefixes
            )
        else:
            self._channel_filter = channel_filter

        self._rate_limit_delay = self._settings.collector.slack_rate_limit_delay
        self._user_cache: dict[str, str] = {}

        logger.info(
            "slack_collector_initialized",
            rate_limit_delay=self._rate_limit_delay,
            blocked_prefixes=self._channel_filter.blocked_prefixes,
        )

    @property
    def client(self) -> AsyncWebClient:
        """Get the Slack async web client."""
        return self._client

    async def _rate_limit_wait(self) -> None:
        """Wait for the configured rate limit delay between API requests."""
        if self._rate_limit_delay > 0:
            await asyncio.sleep(self._rate_limit_delay)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((SlackApiError, TimeoutError)),
        before_sleep=before_sleep_log(logger, structlog.stdlib.logging.WARNING),
        reraise=True,
    )
    async def _fetch_conversations_list(
        self,
        cursor: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Fetch a page of conversations the bot is a member of.

        Args:
            cursor: Pagination cursor for fetching next page.
            limit: Maximum number of channels to return per page.

        Returns:
            The API response containing channel list and pagination info.

        Raises:
            SlackApiError: If the API request fails after retries.
        """
        await self._rate_limit_wait()

        params: dict[str, Any] = {
            "types": "public_channel,private_channel",
            "exclude_archived": False,  # Let filter handle this
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor

        response = await self._client.conversations_list(**params)
        return response.data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((SlackApiError, TimeoutError)),
        before_sleep=before_sleep_log(logger, structlog.stdlib.logging.WARNING),
        reraise=True,
    )
    async def _fetch_conversations_history(
        self,
        channel_id: str,
        oldest: str | None = None,
        latest: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Fetch a page of messages from a channel.

        Args:
            channel_id: The ID of the channel to fetch messages from.
            oldest: Only messages after this Unix timestamp.
            latest: Only messages before this Unix timestamp.
            cursor: Pagination cursor for fetching next page.
            limit: Maximum number of messages to return per page.

        Returns:
            The API response containing messages and pagination info.

        Raises:
            SlackApiError: If the API request fails after retries.
        """
        await self._rate_limit_wait()

        params: dict[str, Any] = {
            "channel": channel_id,
            "limit": limit,
        }
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest
        if cursor:
            params["cursor"] = cursor

        response = await self._client.conversations_history(**params)
        return response.data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((SlackApiError, TimeoutError)),
        before_sleep=before_sleep_log(logger, structlog.stdlib.logging.WARNING),
        reraise=True,
    )
    async def _fetch_conversations_replies(
        self,
        channel_id: str,
        thread_ts: str,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Fetch replies in a thread.

        Args:
            channel_id: The ID of the channel containing the thread.
            thread_ts: The timestamp of the parent message.
            cursor: Pagination cursor for fetching next page.
            limit: Maximum number of replies to return per page.

        Returns:
            The API response containing thread messages and pagination info.

        Raises:
            SlackApiError: If the API request fails after retries.
        """
        await self._rate_limit_wait()

        params: dict[str, Any] = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor

        response = await self._client.conversations_replies(**params)
        return response.data

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((SlackApiError, TimeoutError)),
        before_sleep=before_sleep_log(logger, structlog.stdlib.logging.WARNING),
        reraise=True,
    )
    async def _fetch_user_info(self, user_id: str) -> dict[str, Any]:
        """Fetch user information.

        Args:
            user_id: The Slack user ID.

        Returns:
            The API response containing user information.

        Raises:
            SlackApiError: If the API request fails after retries.
        """
        await self._rate_limit_wait()

        response = await self._client.users_info(user=user_id)
        return response.data

    async def _get_user_name(self, user_id: str) -> str:
        """Get the display name for a user, with caching.

        Args:
            user_id: The Slack user ID.

        Returns:
            The user's display name, or the user ID if lookup fails.
        """
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        try:
            data = await self._fetch_user_info(user_id)
            user = data.get("user", {})
            # Prefer display_name, fall back to real_name, then name
            name = (
                user.get("profile", {}).get("display_name")
                or user.get("real_name")
                or user.get("name")
                or user_id
            )
            self._user_cache[user_id] = name
            return name
        except SlackApiError as e:
            logger.warning(
                "user_info_fetch_failed",
                user_id=user_id,
                error=str(e),
            )
            self._user_cache[user_id] = user_id
            return user_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((SlackApiError, TimeoutError)),
        before_sleep=before_sleep_log(logger, structlog.stdlib.logging.WARNING),
        reraise=True,
    )
    async def _fetch_users_list(
        self,
        cursor: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Fetch a page of workspace users.

        Args:
            cursor: Pagination cursor for fetching next page.
            limit: Maximum number of users to return per page.

        Returns:
            The API response containing user list and pagination info.

        Raises:
            SlackApiError: If the API request fails after retries.
        """
        await self._rate_limit_wait()

        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        response = await self._client.users_list(**params)
        return response.data

    async def list_users(
        self,
        include_bots: bool = False,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """List all valid users in the Slack workspace.

        This method fetches all users and filters out deleted users, bots,
        and app users by default.

        Args:
            include_bots: If True, include bot users. Defaults to False.
            include_deleted: If True, include deleted users. Defaults to False.

        Returns:
            A list of user dictionaries for valid users.

        Raises:
            SlackApiError: If the API request fails after retries.
        """
        users: list[dict[str, Any]] = []
        cursor: str | None = None

        logger.info("listing_slack_users")

        while True:
            try:
                data = await self._fetch_users_list(cursor=cursor)
            except SlackApiError as e:
                logger.error(
                    "users_list_failed",
                    error=str(e),
                    error_code=e.response.get("error") if e.response else None,
                )
                raise

            members = data.get("members", [])

            for user in members:
                # Skip deleted users unless explicitly requested
                if not include_deleted and user.get("deleted", False):
                    logger.debug(
                        "user_skipped_deleted",
                        user_id=user.get("id"),
                        user_name=user.get("name"),
                    )
                    continue

                # Skip bot users unless explicitly requested
                if not include_bots and user.get("is_bot", False):
                    logger.debug(
                        "user_skipped_bot",
                        user_id=user.get("id"),
                        user_name=user.get("name"),
                    )
                    continue

                # Skip app users unless explicitly requested
                if not include_bots and user.get("is_app_user", False):
                    logger.debug(
                        "user_skipped_app_user",
                        user_id=user.get("id"),
                        user_name=user.get("name"),
                    )
                    continue

                users.append(user)

            # Check for more pages
            response_metadata = data.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        logger.info(
            "slack_users_listed",
            total_users=len(users),
        )

        return users

    async def list_sources(self) -> list[DataSource]:
        """List all Slack channels the bot is a member of that pass security filters.

        This method fetches all channels the bot has access to, applies security
        filtering (excluding archived, external shared, and blocked prefix channels),
        and returns only channels where the bot is a member.

        Returns:
            A list of DataSource objects representing accessible Slack channels.

        Raises:
            SlackApiError: If the API request fails after retries.
            ConnectionError: If unable to connect to the Slack API.
        """
        channels: list[DataSource] = []
        cursor: str | None = None

        logger.info("listing_slack_channels")

        while True:
            try:
                data = await self._fetch_conversations_list(cursor=cursor)
            except SlackApiError as e:
                logger.error(
                    "conversations_list_failed",
                    error=str(e),
                    error_code=e.response.get("error") if e.response else None,
                )
                raise

            raw_channels = data.get("channels", [])

            # Apply security filtering
            filtered_channels = self._channel_filter.filter_channels(raw_channels)

            for channel in filtered_channels:
                # Only include channels where bot is a member
                if not channel.get("is_member", False):
                    logger.debug(
                        "channel_skipped_not_member",
                        channel_id=channel.get("id"),
                        channel_name=channel.get("name"),
                    )
                    continue

                channels.append(
                    DataSource(
                        source_type=SourceType.SLACK,
                        source_id=channel["id"],
                        name=channel.get("name", ""),
                        description=channel.get("topic", {}).get("value", "")
                        or channel.get("purpose", {}).get("value", ""),
                        is_accessible=True,
                        member_count=channel.get("num_members"),
                        metadata={
                            "is_private": channel.get("is_private", False),
                            "is_archived": channel.get("is_archived", False),
                            "created": channel.get("created"),
                            "creator": channel.get("creator"),
                        },
                    )
                )

            # Check for more pages
            response_metadata = data.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        logger.info(
            "slack_channels_listed",
            total_channels=len(channels),
        )

        return channels

    def _message_to_raw_item(
        self,
        message: dict[str, Any],
        channel_id: str,
        channel_name: str,
        author_name: str,
    ) -> RawDataItem:
        """Convert a Slack message to a RawDataItem.

        Args:
            message: The raw Slack message data.
            channel_id: The ID of the channel containing the message.
            channel_name: The name of the channel.
            author_name: The resolved display name of the message author.

        Returns:
            A RawDataItem representing the message.
        """
        ts = message.get("ts", "")
        thread_ts = message.get("thread_ts")

        # Determine if this is a reply (has thread_ts and it's different from ts)
        is_reply = thread_ts is not None and thread_ts != ts
        parent_id = thread_ts if is_reply else None

        # Parse timestamp
        try:
            timestamp = datetime.fromtimestamp(float(ts), tz=UTC)
        except (ValueError, TypeError):
            timestamp = datetime.now(tz=UTC)

        return RawDataItem(
            source_type=SourceType.SLACK,
            source_id=ts,
            source_name=channel_name,
            group_id=channel_id,
            content=message.get("text", ""),
            author_id=message.get("user", "unknown"),
            author_name=author_name,
            timestamp=timestamp,
            parent_id=parent_id,
            thread_ts=thread_ts,
            reply_count=message.get("reply_count", 0),
            metadata={
                "subtype": message.get("subtype"),
                "reactions": message.get("reactions", []),
                "attachments": message.get("attachments", []),
                "files": [
                    {
                        "id": f.get("id"),
                        "name": f.get("name"),
                        "mimetype": f.get("mimetype"),
                    }
                    for f in message.get("files", [])
                ],
                "blocks": message.get("blocks", []),
            },
            raw_data=message,
        )

    async def _collect_thread_replies(
        self,
        channel_id: str,
        channel_name: str,
        thread_ts: str,
        config: CollectionConfig,
    ) -> AsyncGenerator[RawDataItem, None]:
        """Collect all replies in a thread.

        Args:
            channel_id: The ID of the channel containing the thread.
            channel_name: The name of the channel.
            thread_ts: The timestamp of the parent message.
            config: Collection configuration.

        Yields:
            RawDataItem objects for each reply in the thread.
        """
        cursor: str | None = None

        while True:
            try:
                data = await self._fetch_conversations_replies(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    cursor=cursor,
                    limit=config.batch_size,
                )
            except SlackApiError as e:
                logger.error(
                    "thread_replies_fetch_failed",
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    error=str(e),
                )
                self._stats.errors.append(
                    f"Thread replies fetch failed for {channel_id}/{thread_ts}: {e}"
                )
                return

            messages = data.get("messages", [])

            for message in messages:
                # Skip the parent message (it has the same ts as thread_ts)
                if message.get("ts") == thread_ts:
                    continue

                # Apply time filters to replies
                try:
                    msg_ts = float(message.get("ts", "0"))
                    msg_time = datetime.fromtimestamp(msg_ts, tz=UTC)

                    if config.start_time and msg_time < config.start_time:
                        continue
                    if config.end_time and msg_time >= config.end_time:
                        continue
                except (ValueError, TypeError):
                    pass

                # Resolve author name
                user_id = message.get("user", "unknown")
                author_name = await self._get_user_name(user_id)

                yield self._message_to_raw_item(
                    message=message,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    author_name=author_name,
                )

            # Check for more pages
            response_metadata = data.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

    async def _collect_channel_messages(
        self,
        channel_id: str,
        channel_name: str,
        config: CollectionConfig,
    ) -> AsyncGenerator[RawDataItem, None]:
        """Collect messages from a single channel.

        Args:
            channel_id: The ID of the channel to collect from.
            channel_name: The name of the channel.
            config: Collection configuration.

        Yields:
            RawDataItem objects for each message and optionally thread replies.
        """
        cursor: str | None = None
        items_collected = 0

        # Convert datetime to Slack timestamp format
        oldest = str(config.start_time.timestamp()) if config.start_time else None
        latest = str(config.end_time.timestamp()) if config.end_time else None

        logger.info(
            "collecting_channel_messages",
            channel_id=channel_id,
            channel_name=channel_name,
            oldest=oldest,
            latest=latest,
        )

        while True:
            try:
                data = await self._fetch_conversations_history(
                    channel_id=channel_id,
                    oldest=oldest,
                    latest=latest,
                    cursor=cursor,
                    limit=config.batch_size,
                )
            except SlackApiError as e:
                logger.error(
                    "channel_history_fetch_failed",
                    channel_id=channel_id,
                    channel_name=channel_name,
                    error=str(e),
                )
                self._stats.errors.append(f"Channel history fetch failed for {channel_name}: {e}")
                return

            messages = data.get("messages", [])

            for message in messages:
                # Check max items limit
                if config.max_items and items_collected >= config.max_items:
                    logger.info(
                        "max_items_reached",
                        channel_id=channel_id,
                        max_items=config.max_items,
                    )
                    return

                # Resolve author name
                user_id = message.get("user", "unknown")
                author_name = await self._get_user_name(user_id)

                # Yield the main message
                yield self._message_to_raw_item(
                    message=message,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    author_name=author_name,
                )
                items_collected += 1

                # Collect thread replies if configured
                if (
                    config.include_threads
                    and config.include_replies
                    and message.get("reply_count", 0) > 0
                ):
                    thread_ts = message.get("thread_ts") or message.get("ts")
                    async for reply in self._collect_thread_replies(
                        channel_id=channel_id,
                        channel_name=channel_name,
                        thread_ts=thread_ts,
                        config=config,
                    ):
                        if config.max_items and items_collected >= config.max_items:
                            return
                        yield reply
                        items_collected += 1

            # Check for more pages
            response_metadata = data.get("response_metadata", {})
            cursor = response_metadata.get("next_cursor")
            if not cursor:
                break

        logger.info(
            "channel_collection_complete",
            channel_id=channel_id,
            channel_name=channel_name,
            items_collected=items_collected,
        )

    async def collect(self, config: CollectionConfig) -> AsyncGenerator[RawDataItem, None]:
        """Collect messages from Slack channels.

        This method fetches messages from the specified channels (or all accessible
        channels if none specified), including thread replies if configured.

        Args:
            config: Configuration for this collection run. If source_ids is empty,
                collects from all accessible channels.

        Yields:
            RawDataItem objects representing Slack messages.

        Raises:
            SlackApiError: If API requests fail after retries.
            ValueError: If the configuration is invalid.

        Example:
            ```python
            config = CollectionConfig(
                start_time=datetime(2024, 1, 1),
                include_threads=True,
                batch_size=100
            )
            async for item in collector.collect(config):
                process_item(item)
            ```
        """
        self.reset_stats()
        self._stats.start_time = datetime.now(tz=UTC)

        logger.info(
            "starting_slack_collection",
            source_ids=config.source_ids,
            start_time=config.start_time.isoformat() if config.start_time else None,
            end_time=config.end_time.isoformat() if config.end_time else None,
            include_threads=config.include_threads,
        )

        # Determine which channels to collect from
        if config.source_ids:
            # Get channel info for specified IDs
            all_sources = await self.list_sources()
            source_map = {s.source_id: s for s in all_sources}
            channels_to_collect = [
                (sid, source_map[sid].name) for sid in config.source_ids if sid in source_map
            ]

            # Warn about missing channels
            missing = set(config.source_ids) - set(source_map.keys())
            if missing:
                logger.warning(
                    "channels_not_found_or_not_accessible",
                    missing_channel_ids=list(missing),
                )
        else:
            # Collect from all accessible channels
            sources = await self.list_sources()
            channels_to_collect = [(s.source_id, s.name) for s in sources]

        logger.info(
            "channels_to_collect",
            count=len(channels_to_collect),
            channels=[name for _, name in channels_to_collect],
        )

        # Collect from each channel
        for channel_id, channel_name in channels_to_collect:
            async for item in self._collect_channel_messages(
                channel_id=channel_id,
                channel_name=channel_name,
                config=config,
            ):
                self._stats.items_collected += 1
                yield item

            self._stats.sources_processed += 1

        self._stats.end_time = datetime.now(tz=UTC)

        logger.info(
            "slack_collection_complete",
            sources_processed=self._stats.sources_processed,
            items_collected=self._stats.items_collected,
            errors=len(self._stats.errors),
            duration_seconds=self._stats.duration_seconds,
        )

    async def validate_connection(self) -> bool:
        """Validate that the Slack connection is working.

        Performs an auth.test API call to verify the bot token is valid.

        Returns:
            True if the connection is valid, False otherwise.
        """
        try:
            await self._rate_limit_wait()
            response = await self._client.auth_test()
            if response.get("ok"):
                logger.info(
                    "slack_connection_valid",
                    team=response.get("team"),
                    user=response.get("user"),
                    bot_id=response.get("bot_id"),
                )
                return True
            return False
        except SlackApiError as e:
            logger.error(
                "slack_connection_invalid",
                error=str(e),
            )
            return False


# Export public API
__all__ = ["SlackCollector"]
