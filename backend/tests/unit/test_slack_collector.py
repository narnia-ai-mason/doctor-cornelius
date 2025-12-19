"""Unit tests for SlackCollector.

Tests cover:
1. Channel list filtering (archived, external, shared channels excluded)
2. User list filtering (valid users only)
3. Thread and reply collection for specific date ranges
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doctor_cornelius.collectors.base import CollectionConfig, DataSource, SourceType
from doctor_cornelius.collectors.slack_collector import SlackCollector
from doctor_cornelius.security.filters import SlackChannelFilter
from tests.conftest import create_slack_response


class TestChannelListFiltering:
    """Tests for channel list filtering functionality.

    Verifies that:
    - Public and private channels accessible to bot are included
    - Archived channels are excluded
    - Externally shared channels are excluded
    - Shared channels (across workspaces) are excluded
    - Channels with blocked prefixes are excluded
    - Channels where bot is not a member are excluded
    """

    @pytest.mark.asyncio
    async def test_list_sources_returns_only_valid_channels(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
        sample_channels: list[dict[str, Any]],
    ) -> None:
        """Test that list_sources returns only valid channels that pass filters."""
        # Setup mock response
        mock_slack_client.conversations_list.return_value = create_slack_response(
            {
                "channels": sample_channels,
                "response_metadata": {"next_cursor": ""},
            }
        )

        # Execute
        sources = await slack_collector.list_sources()

        # Verify
        # Only C001 (general) and C002 (engineering) should be included
        assert len(sources) == 2

        source_ids = {s.source_id for s in sources}
        assert source_ids == {"C001", "C002"}

        # Verify source details
        general = next(s for s in sources if s.source_id == "C001")
        assert general.name == "general"
        assert general.source_type == SourceType.SLACK
        assert general.is_accessible is True
        assert general.metadata["is_private"] is False

        engineering = next(s for s in sources if s.source_id == "C002")
        assert engineering.name == "engineering"
        assert engineering.metadata["is_private"] is True

    @pytest.mark.asyncio
    async def test_archived_channels_are_excluded(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that archived channels are excluded from the list."""
        channels = [
            {
                "id": "C001",
                "name": "active-channel",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
            {
                "id": "C002",
                "name": "archived-channel",
                "is_archived": True,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {"channels": channels, "response_metadata": {"next_cursor": ""}}
        )

        sources = await slack_collector.list_sources()

        assert len(sources) == 1
        assert sources[0].source_id == "C001"
        assert sources[0].name == "active-channel"

    @pytest.mark.asyncio
    async def test_externally_shared_channels_are_excluded(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that externally shared channels (is_ext_shared) are excluded."""
        channels = [
            {
                "id": "C001",
                "name": "internal-channel",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
            {
                "id": "C002",
                "name": "external-partner-channel",
                "is_archived": False,
                "is_ext_shared": True,
                "is_shared": False,
                "is_member": True,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {"channels": channels, "response_metadata": {"next_cursor": ""}}
        )

        sources = await slack_collector.list_sources()

        assert len(sources) == 1
        assert sources[0].source_id == "C001"

    @pytest.mark.asyncio
    async def test_shared_channels_are_excluded(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that shared channels (across workspaces) are excluded."""
        channels = [
            {
                "id": "C001",
                "name": "internal-only",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
            {
                "id": "C002",
                "name": "cross-workspace",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": True,
                "is_member": True,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {"channels": channels, "response_metadata": {"next_cursor": ""}}
        )

        sources = await slack_collector.list_sources()

        assert len(sources) == 1
        assert sources[0].source_id == "C001"

    @pytest.mark.asyncio
    async def test_channels_with_blocked_prefix_are_excluded(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that channels with blocked prefixes are excluded."""
        channels = [
            {
                "id": "C001",
                "name": "engineering",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
            {
                "id": "C002",
                "name": "external-vendors",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
            {
                "id": "C003",
                "name": "guest-access",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {"channels": channels, "response_metadata": {"next_cursor": ""}}
        )

        sources = await slack_collector.list_sources()

        assert len(sources) == 1
        assert sources[0].source_id == "C001"
        assert sources[0].name == "engineering"

    @pytest.mark.asyncio
    async def test_channels_where_bot_is_not_member_are_excluded(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that channels where bot is not a member are excluded."""
        channels = [
            {
                "id": "C001",
                "name": "bot-is-member",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            },
            {
                "id": "C002",
                "name": "bot-not-member",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": False,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {"channels": channels, "response_metadata": {"next_cursor": ""}}
        )

        sources = await slack_collector.list_sources()

        assert len(sources) == 1
        assert sources[0].source_id == "C001"

    @pytest.mark.asyncio
    async def test_pagination_handles_multiple_pages(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that pagination correctly fetches all pages of channels."""
        # First page
        page1_channels = [
            {
                "id": "C001",
                "name": "channel-1",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            }
        ]
        # Second page
        page2_channels = [
            {
                "id": "C002",
                "name": "channel-2",
                "is_archived": False,
                "is_ext_shared": False,
                "is_shared": False,
                "is_member": True,
            }
        ]

        # Mock pagination: first call returns cursor, second call returns empty cursor
        mock_slack_client.conversations_list.side_effect = [
            create_slack_response(
                {
                    "channels": page1_channels,
                    "response_metadata": {"next_cursor": "next_page_cursor"},
                }
            ),
            create_slack_response(
                {
                    "channels": page2_channels,
                    "response_metadata": {"next_cursor": ""},
                }
            ),
        ]

        sources = await slack_collector.list_sources()

        assert len(sources) == 2
        assert {s.source_id for s in sources} == {"C001", "C002"}
        assert mock_slack_client.conversations_list.call_count == 2


class TestSlackChannelFilter:
    """Direct tests for SlackChannelFilter."""

    def test_should_include_valid_channel(self, channel_filter: SlackChannelFilter) -> None:
        """Test that valid channels pass the filter."""
        channel = {
            "id": "C001",
            "name": "engineering",
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
        }
        assert channel_filter.should_include(channel) is True

    def test_should_exclude_archived_channel(self, channel_filter: SlackChannelFilter) -> None:
        """Test that archived channels are excluded."""
        channel = {
            "id": "C001",
            "name": "old-project",
            "is_archived": True,
            "is_ext_shared": False,
            "is_shared": False,
        }
        assert channel_filter.should_include(channel) is False

    def test_should_exclude_ext_shared_channel(self, channel_filter: SlackChannelFilter) -> None:
        """Test that externally shared channels are excluded."""
        channel = {
            "id": "C001",
            "name": "partner-channel",
            "is_archived": False,
            "is_ext_shared": True,
            "is_shared": False,
        }
        assert channel_filter.should_include(channel) is False

    def test_should_exclude_shared_channel(self, channel_filter: SlackChannelFilter) -> None:
        """Test that shared channels are excluded."""
        channel = {
            "id": "C001",
            "name": "cross-workspace",
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": True,
        }
        assert channel_filter.should_include(channel) is False

    def test_should_exclude_blocked_prefix(self, channel_filter: SlackChannelFilter) -> None:
        """Test that channels with blocked prefixes are excluded."""
        external_channel = {
            "id": "C001",
            "name": "external-vendors",
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
        }
        guest_channel = {
            "id": "C002",
            "name": "guest-support",
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
        }
        assert channel_filter.should_include(external_channel) is False
        assert channel_filter.should_include(guest_channel) is False

    def test_filter_channels_returns_only_valid(
        self, channel_filter: SlackChannelFilter, sample_channels: list[dict[str, Any]]
    ) -> None:
        """Test that filter_channels returns only valid channels."""
        filtered = channel_filter.filter_channels(sample_channels)

        # Should only include C001, C002, C008 (before is_member check)
        # C001: general (valid)
        # C002: engineering (valid)
        # C003: old-project (archived)
        # C004: partner-collab (ext_shared)
        # C005: cross-team (shared)
        # C006: external-vendors (blocked prefix)
        # C007: guest-support (blocked prefix)
        # C008: random (valid, but not member - handled by collector)

        valid_ids = {ch["id"] for ch in filtered}
        assert "C001" in valid_ids  # general
        assert "C002" in valid_ids  # engineering
        assert "C003" not in valid_ids  # archived
        assert "C004" not in valid_ids  # ext_shared
        assert "C005" not in valid_ids  # shared
        assert "C006" not in valid_ids  # external- prefix
        assert "C007" not in valid_ids  # guest- prefix
        assert "C008" in valid_ids  # random (filter doesn't check membership)

    def test_get_exclusion_reason(self, channel_filter: SlackChannelFilter) -> None:
        """Test that get_exclusion_reason returns correct reasons."""
        archived = {"name": "old", "is_archived": True}
        assert channel_filter.get_exclusion_reason(archived) == "Channel is archived"

        ext_shared = {"name": "partner", "is_ext_shared": True}
        assert (
            channel_filter.get_exclusion_reason(ext_shared)
            == "Channel is externally shared with other organizations"
        )

        shared = {"name": "cross", "is_shared": True}
        assert channel_filter.get_exclusion_reason(shared) == "Channel is shared across workspaces"

        blocked = {"name": "external-test"}
        assert "blocked prefix" in channel_filter.get_exclusion_reason(blocked)

        valid = {"name": "engineering"}
        assert channel_filter.get_exclusion_reason(valid) is None


class TestUserListFiltering:
    """Tests for user list filtering functionality.

    Verifies that:
    - Active users are included
    - Deleted users are excluded
    - Bot users are excluded
    - App users are excluded
    """

    @pytest.mark.asyncio
    async def test_get_user_name_returns_display_name(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that _get_user_name returns the display name."""
        mock_slack_client.users_info.return_value = create_slack_response(
            {
                "user": {
                    "id": "U001",
                    "name": "john.doe",
                    "real_name": "John Doe",
                    "profile": {
                        "display_name": "John",
                    },
                }
            }
        )

        name = await slack_collector._get_user_name("U001")

        assert name == "John"
        mock_slack_client.users_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_name_falls_back_to_real_name(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that _get_user_name falls back to real_name if display_name is empty."""
        mock_slack_client.users_info.return_value = create_slack_response(
            {
                "user": {
                    "id": "U001",
                    "name": "john.doe",
                    "real_name": "John Doe",
                    "profile": {
                        "display_name": "",
                    },
                }
            }
        )

        name = await slack_collector._get_user_name("U001")

        assert name == "John Doe"

    @pytest.mark.asyncio
    async def test_get_user_name_caches_results(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that user names are cached to avoid repeated API calls."""
        mock_slack_client.users_info.return_value = create_slack_response(
            {
                "user": {
                    "id": "U001",
                    "profile": {"display_name": "John"},
                }
            }
        )

        # First call
        name1 = await slack_collector._get_user_name("U001")
        # Second call (should use cache)
        name2 = await slack_collector._get_user_name("U001")

        assert name1 == name2 == "John"
        # Should only call API once
        assert mock_slack_client.users_info.call_count == 1

    @pytest.mark.asyncio
    async def test_list_users_returns_only_valid_users(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
        sample_users: list[dict[str, Any]],
    ) -> None:
        """Test that list_users returns only valid (non-deleted, non-bot) users."""
        mock_slack_client.users_list.return_value = create_slack_response(
            {
                "members": sample_users,
                "response_metadata": {"next_cursor": ""},
            }
        )

        users = await slack_collector.list_users()

        # Should include: U001 (John), U002 (Jane)
        # Should exclude: U003 (deleted), U004 (bot), U005 (app_user)
        # U006 (restricted guest) is still valid
        valid_user_ids = {u["id"] for u in users}

        assert "U001" in valid_user_ids
        assert "U002" in valid_user_ids
        assert "U003" not in valid_user_ids  # deleted
        assert "U004" not in valid_user_ids  # bot
        assert "U005" not in valid_user_ids  # app_user
        # Guest users might be included depending on requirements

    @pytest.mark.asyncio
    async def test_list_users_excludes_deleted_users(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that deleted users are excluded from the list."""
        users = [
            {
                "id": "U001",
                "name": "active",
                "deleted": False,
                "is_bot": False,
                "is_app_user": False,
            },
            {
                "id": "U002",
                "name": "deleted",
                "deleted": True,
                "is_bot": False,
                "is_app_user": False,
            },
        ]

        mock_slack_client.users_list.return_value = create_slack_response(
            {"members": users, "response_metadata": {"next_cursor": ""}}
        )

        result = await slack_collector.list_users()

        assert len(result) == 1
        assert result[0]["id"] == "U001"

    @pytest.mark.asyncio
    async def test_list_users_excludes_bot_users(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that bot users are excluded from the list."""
        users = [
            {
                "id": "U001",
                "name": "human",
                "deleted": False,
                "is_bot": False,
                "is_app_user": False,
            },
            {"id": "U002", "name": "bot", "deleted": False, "is_bot": True, "is_app_user": False},
        ]

        mock_slack_client.users_list.return_value = create_slack_response(
            {"members": users, "response_metadata": {"next_cursor": ""}}
        )

        result = await slack_collector.list_users()

        assert len(result) == 1
        assert result[0]["id"] == "U001"

    @pytest.mark.asyncio
    async def test_list_users_excludes_app_users(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that app users are excluded from the list."""
        users = [
            {
                "id": "U001",
                "name": "human",
                "deleted": False,
                "is_bot": False,
                "is_app_user": False,
            },
            {"id": "U002", "name": "app", "deleted": False, "is_bot": False, "is_app_user": True},
        ]

        mock_slack_client.users_list.return_value = create_slack_response(
            {"members": users, "response_metadata": {"next_cursor": ""}}
        )

        result = await slack_collector.list_users()

        assert len(result) == 1
        assert result[0]["id"] == "U001"

    @pytest.mark.asyncio
    async def test_list_users_handles_pagination(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that list_users handles pagination correctly."""
        page1 = [
            {"id": "U001", "name": "user1", "deleted": False, "is_bot": False, "is_app_user": False}
        ]
        page2 = [
            {"id": "U002", "name": "user2", "deleted": False, "is_bot": False, "is_app_user": False}
        ]

        mock_slack_client.users_list.side_effect = [
            create_slack_response(
                {"members": page1, "response_metadata": {"next_cursor": "cursor123"}}
            ),
            create_slack_response({"members": page2, "response_metadata": {"next_cursor": ""}}),
        ]

        result = await slack_collector.list_users()

        assert len(result) == 2
        assert {u["id"] for u in result} == {"U001", "U002"}
        assert mock_slack_client.users_list.call_count == 2


class TestThreadAndReplyCollection:
    """Tests for thread and reply collection for specific date ranges.

    Verifies that:
    - Messages within date range are collected
    - Thread replies are collected when configured
    - Parent message is not duplicated in thread replies
    - Date filtering is applied to replies
    """

    @pytest.mark.asyncio
    async def test_collect_messages_for_specific_date(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test collecting messages for a specific date (one day)."""
        # 2024-01-15 00:00:00 to 2024-01-16 00:00:00 UTC
        start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 16, 0, 0, 0, tzinfo=UTC)

        # Messages at different times on 2024-01-15
        messages = [
            {
                "type": "message",
                "ts": "1705312800.000001",  # 2024-01-15 10:00:00 UTC
                "user": "U001",
                "text": "Morning message",
            },
            {
                "type": "message",
                "ts": "1705348800.000001",  # 2024-01-15 20:00:00 UTC
                "user": "U002",
                "text": "Evening message",
            },
        ]

        # Setup mocks
        mock_slack_client.conversations_list.return_value = create_slack_response(
            {
                "channels": [
                    {
                        "id": "C001",
                        "name": "general",
                        "is_archived": False,
                        "is_ext_shared": False,
                        "is_shared": False,
                        "is_member": True,
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        )

        mock_slack_client.conversations_history.return_value = create_slack_response(
            {"messages": messages, "response_metadata": {"next_cursor": ""}}
        )

        mock_slack_client.users_info.return_value = create_slack_response(
            {"user": {"id": "U001", "profile": {"display_name": "User"}}}
        )

        # Execute
        config = CollectionConfig(
            source_ids=["C001"],
            start_time=start_time,
            end_time=end_time,
            include_threads=False,
        )

        collected = []
        async for item in slack_collector.collect(config):
            collected.append(item)

        # Verify
        assert len(collected) == 2

        # Verify API was called with correct timestamp parameters
        call_kwargs = mock_slack_client.conversations_history.call_args.kwargs
        assert call_kwargs["oldest"] == str(start_time.timestamp())
        assert call_kwargs["latest"] == str(end_time.timestamp())

    @pytest.mark.asyncio
    async def test_collect_threads_and_replies(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that thread replies are collected when include_threads is True."""
        parent_ts = "1705312800.000001"

        # Parent message with replies
        messages = [
            {
                "type": "message",
                "ts": parent_ts,
                "user": "U001",
                "text": "Thread starter",
                "thread_ts": parent_ts,
                "reply_count": 2,
            },
        ]

        # Thread replies (including parent as first message)
        thread_replies = [
            {
                "type": "message",
                "ts": parent_ts,
                "user": "U001",
                "text": "Thread starter",
                "thread_ts": parent_ts,
            },
            {
                "type": "message",
                "ts": "1705312850.000001",
                "user": "U002",
                "text": "First reply",
                "thread_ts": parent_ts,
            },
            {
                "type": "message",
                "ts": "1705312860.000001",
                "user": "U001",
                "text": "Second reply",
                "thread_ts": parent_ts,
            },
        ]

        # Setup mocks
        mock_slack_client.conversations_list.return_value = create_slack_response(
            {
                "channels": [
                    {
                        "id": "C001",
                        "name": "general",
                        "is_archived": False,
                        "is_ext_shared": False,
                        "is_shared": False,
                        "is_member": True,
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        )

        mock_slack_client.conversations_history.return_value = create_slack_response(
            {"messages": messages, "response_metadata": {"next_cursor": ""}}
        )

        mock_slack_client.conversations_replies.return_value = create_slack_response(
            {"messages": thread_replies, "response_metadata": {"next_cursor": ""}}
        )

        mock_slack_client.users_info.return_value = create_slack_response(
            {"user": {"id": "U001", "profile": {"display_name": "User"}}}
        )

        # Execute
        config = CollectionConfig(
            source_ids=["C001"],
            include_threads=True,
            include_replies=True,
        )

        collected = []
        async for item in slack_collector.collect(config):
            collected.append(item)

        # Verify: should have parent + 2 replies = 3 items
        assert len(collected) == 3

        # Parent message
        parent = collected[0]
        assert parent.content == "Thread starter"
        assert parent.source_id == parent_ts
        assert parent.parent_id is None  # Parent has no parent

        # Replies
        replies = [c for c in collected if c.parent_id is not None]
        assert len(replies) == 2
        assert all(r.thread_ts == parent_ts for r in replies)

    @pytest.mark.asyncio
    async def test_thread_replies_exclude_parent_message(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that parent message is not duplicated when collecting thread replies."""
        parent_ts = "1705312800.000001"

        messages = [
            {
                "type": "message",
                "ts": parent_ts,
                "user": "U001",
                "text": "Parent",
                "thread_ts": parent_ts,
                "reply_count": 1,
            },
        ]

        # Thread API returns parent as first message
        thread_replies = [
            {
                "type": "message",
                "ts": parent_ts,  # Same as parent
                "user": "U001",
                "text": "Parent",
                "thread_ts": parent_ts,
            },
            {
                "type": "message",
                "ts": "1705312850.000001",
                "user": "U002",
                "text": "Reply",
                "thread_ts": parent_ts,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {
                "channels": [
                    {
                        "id": "C001",
                        "name": "ch",
                        "is_archived": False,
                        "is_ext_shared": False,
                        "is_shared": False,
                        "is_member": True,
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        )
        mock_slack_client.conversations_history.return_value = create_slack_response(
            {"messages": messages, "response_metadata": {"next_cursor": ""}}
        )
        mock_slack_client.conversations_replies.return_value = create_slack_response(
            {"messages": thread_replies, "response_metadata": {"next_cursor": ""}}
        )
        mock_slack_client.users_info.return_value = create_slack_response(
            {"user": {"profile": {"display_name": "User"}}}
        )

        config = CollectionConfig(source_ids=["C001"], include_threads=True, include_replies=True)

        collected = []
        async for item in slack_collector.collect(config):
            collected.append(item)

        # Should have exactly 2 items: 1 parent + 1 reply (parent not duplicated from thread)
        assert len(collected) == 2

        # Verify only one item has the parent_ts as source_id
        parent_items = [c for c in collected if c.source_id == parent_ts]
        assert len(parent_items) == 1

    @pytest.mark.asyncio
    async def test_date_filter_applies_to_thread_replies(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that date filtering is applied to thread replies."""
        # Filter for 2024-01-15 only
        start_time = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 16, 0, 0, 0, tzinfo=UTC)

        parent_ts = "1705312800.000001"  # 2024-01-15 10:00:00

        messages = [
            {
                "type": "message",
                "ts": parent_ts,
                "user": "U001",
                "text": "Parent",
                "thread_ts": parent_ts,
                "reply_count": 2,
            },
        ]

        # One reply on 1/15, one reply on 1/16 (should be excluded)
        thread_replies = [
            {
                "type": "message",
                "ts": parent_ts,
                "user": "U001",
                "text": "Parent",
                "thread_ts": parent_ts,
            },
            # Reply on 2024-01-15 15:00:00 - should be included
            {
                "type": "message",
                "ts": "1705330800.000001",
                "user": "U002",
                "text": "Reply on 1/15",
                "thread_ts": parent_ts,
            },
            # Reply on 2024-01-16 10:00:00 - should be excluded
            {
                "type": "message",
                "ts": "1705399200.000001",
                "user": "U002",
                "text": "Reply on 1/16",
                "thread_ts": parent_ts,
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {
                "channels": [
                    {
                        "id": "C001",
                        "name": "ch",
                        "is_archived": False,
                        "is_ext_shared": False,
                        "is_shared": False,
                        "is_member": True,
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        )
        mock_slack_client.conversations_history.return_value = create_slack_response(
            {"messages": messages, "response_metadata": {"next_cursor": ""}}
        )
        mock_slack_client.conversations_replies.return_value = create_slack_response(
            {"messages": thread_replies, "response_metadata": {"next_cursor": ""}}
        )
        mock_slack_client.users_info.return_value = create_slack_response(
            {"user": {"profile": {"display_name": "User"}}}
        )

        config = CollectionConfig(
            source_ids=["C001"],
            start_time=start_time,
            end_time=end_time,
            include_threads=True,
            include_replies=True,
        )

        collected = []
        async for item in slack_collector.collect(config):
            collected.append(item)

        # Should have: 1 parent + 1 reply (on 1/15) = 2 items
        # Reply on 1/16 should be excluded
        assert len(collected) == 2
        assert not any("1/16" in c.content for c in collected)

    @pytest.mark.asyncio
    async def test_collect_without_threads(
        self,
        slack_collector: SlackCollector,
        mock_slack_client: AsyncMock,
    ) -> None:
        """Test that threads are not fetched when include_threads is False."""
        messages = [
            {
                "type": "message",
                "ts": "1705312800.000001",
                "user": "U001",
                "text": "Message with replies",
                "reply_count": 5,  # Has replies but we shouldn't fetch them
            },
        ]

        mock_slack_client.conversations_list.return_value = create_slack_response(
            {
                "channels": [
                    {
                        "id": "C001",
                        "name": "ch",
                        "is_archived": False,
                        "is_ext_shared": False,
                        "is_shared": False,
                        "is_member": True,
                    }
                ],
                "response_metadata": {"next_cursor": ""},
            }
        )
        mock_slack_client.conversations_history.return_value = create_slack_response(
            {"messages": messages, "response_metadata": {"next_cursor": ""}}
        )
        mock_slack_client.users_info.return_value = create_slack_response(
            {"user": {"profile": {"display_name": "User"}}}
        )

        config = CollectionConfig(
            source_ids=["C001"],
            include_threads=False,
        )

        collected = []
        async for item in slack_collector.collect(config):
            collected.append(item)

        # Should only have the parent message
        assert len(collected) == 1
        # conversations_replies should NOT have been called
        mock_slack_client.conversations_replies.assert_not_called()
