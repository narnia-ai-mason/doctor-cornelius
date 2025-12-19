"""Slack Bot handlers for Doctor Cornelius.

This module implements the Slack Bot using slack-bolt with Socket Mode.
It provides handlers for:
- @mention events - Search knowledge base and answer questions
- Direct Messages - Direct conversation with the bot
- Thread replies - Respond in threads when appropriate

Usage:
    from doctor_cornelius.bot.app import create_app, start_socket_mode

    # Create the Bolt app
    app = create_app()

    # Start Socket Mode (blocking)
    await start_socket_mode(app)
"""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import structlog
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

from doctor_cornelius.agent import close_agent_manager, get_agent_manager
from doctor_cornelius.config import Settings, get_settings

if TYPE_CHECKING:
    from slack_bolt.context.ack.async_ack import AsyncAck
    from slack_bolt.context.say.async_say import AsyncSay
    from slack_sdk.web.async_client import AsyncWebClient

logger = structlog.get_logger(__name__)


def format_error_message(error_type: str, details: str | None = None) -> str:
    """Format a user-friendly error message.

    Args:
        error_type: Type of error that occurred.
        details: Optional additional details.

    Returns:
        Formatted error message for Slack.
    """
    messages = {
        "search_failed": (
            "I encountered an issue while searching the knowledge base. "
            "Please try again in a moment."
        ),
        "initialization_failed": (
            "I'm having trouble connecting to my knowledge base. "
            "Please contact an administrator if this persists."
        ),
        "rate_limited": (
            "I'm receiving too many requests right now. Please wait a moment and try again."
        ),
        "empty_query": (
            "I didn't receive a question to answer. "
            "Please mention me with a question, like: `@Doctor Cornelius what is...?`"
        ),
        "unknown": (
            "Something unexpected happened. "
            "Please try again or contact an administrator if this persists."
        ),
    }

    message = messages.get(error_type, messages["unknown"])

    if details:
        message += f"\n\n_Error details: {details}_"

    return message


def extract_question_from_mention(text: str, bot_user_id: str) -> str:
    """Extract the question from a mention text, removing the bot mention.

    Args:
        text: The raw message text containing the mention.
        bot_user_id: The bot's user ID to remove from the text.

    Returns:
        The cleaned question text.
    """
    # Remove the bot mention (format: <@USERID>)
    mention_pattern = rf"<@{bot_user_id}>"
    cleaned = re.sub(mention_pattern, "", text, flags=re.IGNORECASE).strip()

    # Also handle any leading/trailing whitespace or punctuation
    cleaned = cleaned.strip(" ,:")

    return cleaned


async def handle_message(
    message: str,
    settings: Settings | None = None,
) -> str:
    """Process a message using the LangGraph agent.

    The agent will decide whether to search the knowledge base
    based on the content of the message.

    Args:
        message: The user's message to process.
        settings: Application settings.

    Returns:
        The agent's response string.
    """
    log = logger.bind(message_preview=message[:100])

    if not message.strip():
        log.warning("empty_message_received")
        return format_error_message("empty_query")

    try:
        # Get or initialize the agent manager
        agent_manager = await get_agent_manager(settings)

        # Process the message through the agent
        log.info("processing_message_with_agent")
        response = await agent_manager.chat(message)

        log.info("agent_response_generated", response_length=len(response))
        return response

    except RuntimeError as e:
        log.error("agent_initialization_error", error=str(e))
        return format_error_message("initialization_failed")

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate" in error_str:
            log.warning("rate_limited", error=str(e))
            return format_error_message("rate_limited")

        log.error("agent_error", error_type=type(e).__name__, error=str(e))
        return format_error_message("unknown")


def create_app(settings: Settings | None = None) -> AsyncApp:
    """Create and configure the Slack Bolt async app.

    Args:
        settings: Application settings. If not provided, loads from environment.

    Returns:
        Configured AsyncApp instance with all handlers registered.
    """
    if settings is None:
        settings = get_settings()

    log = logger.bind(component="slack_bot")
    log.info("creating_slack_app")

    # Create the Bolt app with Socket Mode configuration
    app = AsyncApp(
        token=settings.slack.bot_token.get_secret_value(),
        signing_secret=settings.slack.signing_secret.get_secret_value(),
    )

    # Store settings in app for access in handlers
    app._doctor_cornelius_settings = settings  # type: ignore[attr-defined]

    @app.event("app_mention")
    async def handle_mention(
        event: dict[str, Any],
        say: AsyncSay,
        client: AsyncWebClient,
        ack: AsyncAck,
    ) -> None:
        """Handle @mention events.

        When the bot is mentioned in a channel, extract the question,
        search the knowledge base, and respond with relevant information.
        """
        await ack()

        channel = event.get("channel", "")
        user = event.get("user", "")
        text = event.get("text", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        # If mentioned in a thread, reply in the thread; otherwise start a new thread
        reply_thread_ts = thread_ts or ts

        mention_log = log.bind(
            event_type="app_mention",
            channel=channel,
            user=user,
            thread_ts=thread_ts,
        )
        mention_log.info("mention_received")

        try:
            # Get bot user ID to extract the question
            auth_response = await client.auth_test()
            bot_user_id = auth_response.get("user_id", "")

            # Extract the question from the mention
            question = extract_question_from_mention(text, bot_user_id)

            if not question:
                mention_log.warning("empty_question_in_mention")
                await say(
                    text=format_error_message("empty_query"),
                    channel=channel,
                    thread_ts=reply_thread_ts,
                )
                return

            # Show typing indicator while processing
            # Note: Slack's "typing" indicator requires a separate API call
            # We'll use reactions as a visual indicator instead
            with suppress(SlackApiError):
                await client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name="hourglass_flowing_sand",
                )

            # Process the message through the agent
            response = await handle_message(
                message=question,
                settings=app._doctor_cornelius_settings,  # type: ignore[attr-defined]
            )

            # Remove the "processing" reaction and add "done" reaction
            try:
                await client.reactions_remove(
                    channel=channel,
                    timestamp=ts,
                    name="hourglass_flowing_sand",
                )
                await client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name="white_check_mark",
                )
            except SlackApiError:
                pass  # Ignore reaction errors

            # Send the response
            await say(
                text=response,
                channel=channel,
                thread_ts=reply_thread_ts,
            )

            mention_log.info("mention_handled_successfully")

        except SlackApiError as e:
            mention_log.error(
                "slack_api_error",
                error_type=type(e).__name__,
                error_code=e.response.get("error") if e.response else None,
            )
            await say(
                text=format_error_message("unknown", details="Slack API error"),
                channel=channel,
                thread_ts=reply_thread_ts,
            )

        except Exception as e:
            mention_log.error(
                "mention_handler_error",
                error_type=type(e).__name__,
                error=str(e),
            )
            with suppress(Exception):
                await say(
                    text=format_error_message("unknown"),
                    channel=channel,
                    thread_ts=reply_thread_ts,
                )

    @app.event("message")
    async def handle_dm(
        event: dict[str, Any],
        say: AsyncSay,
        client: AsyncWebClient,
        ack: AsyncAck,
    ) -> None:
        """Handle direct message events.

        This handler processes DMs to the bot, treating each message
        as a question to search the knowledge base.
        """
        await ack()

        # Get channel info
        channel = event.get("channel", "")
        channel_type = event.get("channel_type", "")
        user = event.get("user", "")
        text = event.get("text", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")
        subtype = event.get("subtype")

        # Only handle DMs (channel_type == "im")
        if channel_type != "im":
            return

        # Ignore bot messages, message_changed, and other subtypes
        if subtype is not None:
            return

        # Ignore messages without text
        if not text:
            return

        # For DMs, reply in thread if the message is in a thread
        reply_thread_ts = thread_ts or ts

        dm_log = log.bind(
            event_type="dm",
            channel=channel,
            user=user,
            thread_ts=thread_ts,
        )
        dm_log.info("dm_received")

        try:
            # Show typing indicator
            with suppress(SlackApiError):
                await client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name="hourglass_flowing_sand",
                )

            # Clean the message (remove bot mention if present, though unlikely in DMs)
            auth_response = await client.auth_test()
            bot_user_id = auth_response.get("user_id", "")
            question = extract_question_from_mention(text, bot_user_id)

            if not question:
                dm_log.warning("empty_dm")
                await say(
                    text="Hello! Ask me anything about the team's knowledge base. "
                    "Just type your question and I'll search for relevant information.",
                    channel=channel,
                    thread_ts=reply_thread_ts,
                )
                return

            # Process the message through the agent
            response = await handle_message(
                message=question,
                settings=app._doctor_cornelius_settings,  # type: ignore[attr-defined]
            )

            # Update reactions
            with suppress(SlackApiError):
                await client.reactions_remove(
                    channel=channel,
                    timestamp=ts,
                    name="hourglass_flowing_sand",
                )
                await client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name="white_check_mark",
                )

            # Send the response
            await say(
                text=response,
                channel=channel,
                thread_ts=reply_thread_ts,
            )

            dm_log.info("dm_handled_successfully")

        except SlackApiError as e:
            dm_log.error(
                "slack_api_error",
                error_type=type(e).__name__,
                error_code=e.response.get("error") if e.response else None,
            )
            await say(
                text=format_error_message("unknown", details="Slack API error"),
                channel=channel,
                thread_ts=reply_thread_ts,
            )

        except Exception as e:
            dm_log.error(
                "dm_handler_error",
                error_type=type(e).__name__,
                error=str(e),
            )
            with suppress(Exception):
                await say(
                    text=format_error_message("unknown"),
                    channel=channel,
                    thread_ts=reply_thread_ts,
                )

    @app.event("app_home_opened")
    async def handle_app_home_opened(
        event: dict[str, Any],
        client: AsyncWebClient,
        ack: AsyncAck,
    ) -> None:
        """Handle App Home tab opened event.

        Displays a welcome view in the App Home tab.
        """
        await ack()

        user = event.get("user", "")

        home_log = log.bind(event_type="app_home_opened", user=user)
        home_log.info("app_home_opened")

        try:
            await client.views_publish(
                user_id=user,
                view={
                    "type": "home",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "Welcome to Doctor Cornelius",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": (
                                    "I'm Doctor Cornelius, the trusted archivist of Narnia Labs. "
                                    "I help you find information from our team's knowledge base."
                                ),
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*How to use me:*",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": (
                                    "1. *Mention me* in any channel: "
                                    "`@Doctor Cornelius what is...?`\n"
                                    "2. *DM me* directly with your question\n"
                                    "3. *Reply in threads* for follow-up questions"
                                ),
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": (
                                        "_I search our temporal knowledge graph to find "
                                        "relevant information from past conversations"
                                        " and documents._"
                                    ),
                                }
                            ],
                        },
                    ],
                },
            )
            home_log.info("app_home_view_published")

        except SlackApiError as e:
            home_log.error(
                "app_home_publish_error",
                error_type=type(e).__name__,
                error_code=e.response.get("error") if e.response else None,
            )

    log.info("slack_app_created")
    return app


async def start_socket_mode(
    app: AsyncApp,
    settings: Settings | None = None,
) -> None:
    """Start the Slack app in Socket Mode.

    This function starts the WebSocket connection to Slack and
    begins processing events. It runs indefinitely until interrupted.

    Args:
        app: The configured AsyncApp instance.
        settings: Application settings. If not provided, loads from environment.
    """
    if settings is None:
        settings = get_settings()

    log = logger.bind(component="socket_mode")
    log.info("starting_socket_mode")

    try:
        handler = AsyncSocketModeHandler(
            app=app,
            app_token=settings.slack.app_token.get_secret_value(),
        )

        log.info("socket_mode_handler_created")

        # Start the handler (this blocks until interrupted)
        await handler.start_async()

    except KeyboardInterrupt:
        log.info("socket_mode_interrupted")
        raise

    except Exception as e:
        log.error(
            "socket_mode_error",
            error_type=type(e).__name__,
            error=str(e),
        )
        raise

    finally:
        # Clean up the agent manager
        await close_agent_manager()
        log.info("socket_mode_shutdown_complete")


async def run_bot() -> None:
    """Main entry point to run the Slack bot.

    Creates the app and starts Socket Mode. This function
    runs indefinitely until interrupted.
    """
    log = logger.bind(component="bot_main")
    log.info("bot_starting")

    try:
        settings = get_settings()
        app = create_app(settings)
        await start_socket_mode(app, settings)

    except KeyboardInterrupt:
        log.info("bot_shutdown_requested")

    except Exception as e:
        log.error(
            "bot_fatal_error",
            error_type=type(e).__name__,
            error=str(e),
        )
        raise

    finally:
        log.info("bot_stopped")


# For running the bot directly
if __name__ == "__main__":
    asyncio.run(run_bot())
