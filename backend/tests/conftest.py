"""Shared test fixtures and configuration for Doctor Cornelius tests."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from slack_sdk.web.async_client import AsyncWebClient

from doctor_cornelius.collectors.slack_collector import SlackCollector
from doctor_cornelius.security.filters import SlackChannelFilter


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.slack.bot_token.get_secret_value.return_value = "xoxb-test-token"
    settings.collector.blocked_channel_prefixes = ["external-", "guest-"]
    settings.collector.slack_rate_limit_delay = 0  # No delay in tests
    return settings


@pytest.fixture
def channel_filter() -> SlackChannelFilter:
    """Create a default channel filter for testing."""
    return SlackChannelFilter(
        blocked_prefixes=["external-", "guest-"],
        allow_archived=False,
        allow_external_shared=False,
        allow_shared=False,
    )


@pytest.fixture
def mock_slack_client() -> AsyncMock:
    """Create a mock Slack async web client."""
    return AsyncMock(spec=AsyncWebClient)


@pytest_asyncio.fixture
async def slack_collector(
    mock_settings: MagicMock,
    channel_filter: SlackChannelFilter,
    mock_slack_client: AsyncMock,
) -> AsyncIterator[SlackCollector]:
    """Create a SlackCollector with mocked dependencies."""
    collector = SlackCollector(
        settings=mock_settings,
        channel_filter=channel_filter,
    )
    # Replace the client with our mock
    collector._client = mock_slack_client
    yield collector


# Sample channel data for testing
@pytest.fixture
def sample_channels() -> list[dict[str, Any]]:
    """Create sample channel data with various types."""
    return [
        # Valid public channel - should be included
        {
            "id": "C001",
            "name": "general",
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
            "is_member": True,
            "num_members": 50,
            "topic": {"value": "General discussion"},
            "purpose": {"value": "Company-wide announcements"},
            "created": 1609459200,
            "creator": "U001",
        },
        # Valid private channel - should be included
        {
            "id": "C002",
            "name": "engineering",
            "is_channel": True,
            "is_private": True,
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
            "is_member": True,
            "num_members": 20,
            "topic": {"value": "Engineering team"},
            "purpose": {"value": "Engineering discussions"},
            "created": 1609459200,
            "creator": "U001",
        },
        # Archived channel - should be EXCLUDED
        {
            "id": "C003",
            "name": "old-project",
            "is_channel": True,
            "is_private": False,
            "is_archived": True,
            "is_ext_shared": False,
            "is_shared": False,
            "is_member": True,
            "num_members": 10,
            "topic": {"value": "Old project"},
            "purpose": {"value": ""},
            "created": 1609459200,
            "creator": "U001",
        },
        # Externally shared channel - should be EXCLUDED
        {
            "id": "C004",
            "name": "partner-collab",
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_ext_shared": True,
            "is_shared": False,
            "is_member": True,
            "num_members": 15,
            "topic": {"value": "Partner collaboration"},
            "purpose": {"value": ""},
            "created": 1609459200,
            "creator": "U001",
        },
        # Shared channel across workspaces - should be EXCLUDED
        {
            "id": "C005",
            "name": "cross-team",
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": True,
            "is_member": True,
            "num_members": 30,
            "topic": {"value": "Cross-team channel"},
            "purpose": {"value": ""},
            "created": 1609459200,
            "creator": "U001",
        },
        # Channel with blocked prefix - should be EXCLUDED
        {
            "id": "C006",
            "name": "external-vendors",
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
            "is_member": True,
            "num_members": 5,
            "topic": {"value": "External vendors"},
            "purpose": {"value": ""},
            "created": 1609459200,
            "creator": "U001",
        },
        # Another channel with blocked prefix - should be EXCLUDED
        {
            "id": "C007",
            "name": "guest-support",
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
            "is_member": True,
            "num_members": 3,
            "topic": {"value": "Guest support"},
            "purpose": {"value": ""},
            "created": 1609459200,
            "creator": "U001",
        },
        # Valid channel but bot is NOT a member - should be EXCLUDED
        {
            "id": "C008",
            "name": "random",
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_ext_shared": False,
            "is_shared": False,
            "is_member": False,  # Bot is not a member
            "num_members": 40,
            "topic": {"value": "Random stuff"},
            "purpose": {"value": ""},
            "created": 1609459200,
            "creator": "U001",
        },
    ]


@pytest.fixture
def sample_users() -> list[dict[str, Any]]:
    """Create sample user data with various types."""
    return [
        # Valid active user
        {
            "id": "U001",
            "name": "john.doe",
            "real_name": "John Doe",
            "deleted": False,
            "is_bot": False,
            "is_app_user": False,
            "profile": {
                "display_name": "John",
                "email": "john@example.com",
                "image_72": "https://example.com/john.png",
            },
        },
        # Valid active user
        {
            "id": "U002",
            "name": "jane.smith",
            "real_name": "Jane Smith",
            "deleted": False,
            "is_bot": False,
            "is_app_user": False,
            "profile": {
                "display_name": "Jane",
                "email": "jane@example.com",
                "image_72": "https://example.com/jane.png",
            },
        },
        # Deleted user - should be EXCLUDED
        {
            "id": "U003",
            "name": "deleted.user",
            "real_name": "Deleted User",
            "deleted": True,
            "is_bot": False,
            "is_app_user": False,
            "profile": {
                "display_name": "",
                "email": "",
            },
        },
        # Bot user - should be EXCLUDED (unless specifically requested)
        {
            "id": "U004",
            "name": "slack-bot",
            "real_name": "Slack Bot",
            "deleted": False,
            "is_bot": True,
            "is_app_user": False,
            "profile": {
                "display_name": "SlackBot",
            },
        },
        # App user - should be EXCLUDED (unless specifically requested)
        {
            "id": "U005",
            "name": "integration-app",
            "real_name": "Integration App",
            "deleted": False,
            "is_bot": False,
            "is_app_user": True,
            "profile": {
                "display_name": "Integration",
            },
        },
        # Restricted guest user - valid but limited
        {
            "id": "U006",
            "name": "guest.user",
            "real_name": "Guest User",
            "deleted": False,
            "is_bot": False,
            "is_app_user": False,
            "is_restricted": True,
            "profile": {
                "display_name": "Guest",
                "email": "guest@external.com",
            },
        },
    ]


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    """Create sample message data including threads and replies."""
    # Base timestamp: 2024-01-15 10:00:00 UTC
    base_ts = 1705312800.0

    return [
        # Parent message with thread
        {
            "type": "message",
            "ts": f"{base_ts}.000001",
            "user": "U001",
            "text": "This is a thread starter message",
            "thread_ts": f"{base_ts}.000001",
            "reply_count": 2,
            "reply_users_count": 2,
        },
        # Regular message (no thread)
        {
            "type": "message",
            "ts": f"{base_ts + 100}.000002",
            "user": "U002",
            "text": "This is a standalone message",
        },
        # Another parent message with replies
        {
            "type": "message",
            "ts": f"{base_ts + 200}.000003",
            "user": "U001",
            "text": "Another thread starter",
            "thread_ts": f"{base_ts + 200}.000003",
            "reply_count": 1,
            "reply_users_count": 1,
        },
    ]


@pytest.fixture
def sample_thread_replies() -> list[dict[str, Any]]:
    """Create sample thread reply data."""
    base_ts = 1705312800.0
    thread_ts = f"{base_ts}.000001"

    return [
        # Parent message (returned first in conversations.replies)
        {
            "type": "message",
            "ts": thread_ts,
            "user": "U001",
            "text": "This is a thread starter message",
            "thread_ts": thread_ts,
            "reply_count": 2,
        },
        # First reply
        {
            "type": "message",
            "ts": f"{base_ts + 50}.000001",
            "user": "U002",
            "text": "This is the first reply",
            "thread_ts": thread_ts,
        },
        # Second reply
        {
            "type": "message",
            "ts": f"{base_ts + 60}.000001",
            "user": "U001",
            "text": "This is the second reply",
            "thread_ts": thread_ts,
        },
    ]


def create_slack_response(data: dict[str, Any], ok: bool = True) -> MagicMock:
    """Create a mock Slack API response."""
    response = MagicMock()
    response.data = {"ok": ok, **data}
    response.get = lambda key, default=None: response.data.get(key, default)
    return response
