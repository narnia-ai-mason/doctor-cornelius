"""Agent module for Doctor Cornelius.

This module provides a LangGraph-based ReAct agent that intelligently
decides when to search the knowledge base based on user messages.

Usage:
    from doctor_cornelius.agent import get_agent_manager, close_agent_manager

    # Get or create the agent manager singleton
    manager = await get_agent_manager()

    # Process a message
    response = await manager.chat("What projects is the team working on?")

    # Clean up on shutdown
    await close_agent_manager()
"""

from doctor_cornelius.agent.manager import (
    AgentManager,
    close_agent_manager,
    get_agent_manager,
)

__all__ = [
    "AgentManager",
    "get_agent_manager",
    "close_agent_manager",
]
