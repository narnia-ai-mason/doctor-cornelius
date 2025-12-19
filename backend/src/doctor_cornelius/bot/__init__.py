"""Slack Bot module for Doctor Cornelius.

This module provides the Slack Bot implementation using slack-bolt
with Socket Mode for real-time event handling.
"""

from doctor_cornelius.bot.app import (
    create_app,
    run_bot,
    start_socket_mode,
)

__all__ = [
    "create_app",
    "start_socket_mode",
    "run_bot",
]
