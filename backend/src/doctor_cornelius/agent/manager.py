"""Agent manager for Doctor Cornelius.

This module provides a singleton manager for the LangGraph agent,
handling initialization, lifecycle, and message processing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from langchain_core.messages import HumanMessage

from doctor_cornelius.agent.graph import create_doctor_cornelius_agent
from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.knowledge.graph_client import GraphitiClientManager

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

logger = structlog.get_logger(__name__)


class AgentManager:
    """Managed agent instance with Graphiti integration.

    This class provides:
    - Lazy initialization of the agent and Graphiti client
    - Thread-safe singleton access
    - Clean shutdown handling
    - Simple chat interface for message processing

    Usage:
        manager = AgentManager()
        await manager.initialize()
        try:
            response = await manager.chat("What projects is the team working on?")
        finally:
            await manager.close()

    Or with singleton:
        manager = await get_agent_manager()
        response = await manager.chat("Hello!")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the AgentManager.

        Args:
            settings: Application settings. If not provided, loads from environment.
        """
        self._settings = settings or get_settings()
        self._graphiti_manager: GraphitiClientManager | None = None
        self._agent: CompiledGraph | None = None
        self._initialized = False

        self._log = logger.bind(
            component="agent_manager",
            model=self._settings.gemini.model,
        )

    async def initialize(self) -> None:
        """Initialize the agent and Graphiti client.

        This sets up:
        - GraphitiClientManager for knowledge base operations
        - LangGraph ReAct agent with Gemini model

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            self._log.debug("agent_already_initialized")
            return

        self._log.info("initializing_agent_manager")

        try:
            # Initialize Graphiti client first
            self._graphiti_manager = GraphitiClientManager(self._settings)
            await self._graphiti_manager.initialize()

            # Create the agent with the initialized Graphiti manager
            self._agent = create_doctor_cornelius_agent(
                settings=self._settings,
                graphiti_manager=self._graphiti_manager,
            )

            self._initialized = True
            self._log.info("agent_manager_initialized")

        except Exception as e:
            self._log.error(
                "agent_initialization_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            # Clean up partial initialization
            if self._graphiti_manager is not None:
                await self._graphiti_manager.close()
                self._graphiti_manager = None
            raise RuntimeError(f"Failed to initialize agent: {e}") from e

    async def close(self) -> None:
        """Close the agent and release resources."""
        self._log.info("closing_agent_manager")

        try:
            if self._graphiti_manager is not None:
                await self._graphiti_manager.close()
        except Exception as e:
            self._log.warning(
                "graphiti_close_warning",
                error_type=type(e).__name__,
                error_message=str(e),
            )
        finally:
            self._graphiti_manager = None
            self._agent = None
            self._initialized = False
            self._log.info("agent_manager_closed")

    async def chat(self, message: str) -> str:
        """Process a chat message and return the agent's response.

        The agent will decide whether to search the knowledge base
        based on the content of the message.

        Args:
            message: The user's message to process.

        Returns:
            The agent's response as a string.

        Raises:
            RuntimeError: If the agent is not initialized.
        """
        if not self._initialized or self._agent is None:
            raise RuntimeError("AgentManager is not initialized. Call initialize() first.")

        log = self._log.bind(message_preview=message[:50])
        log.info("processing_chat_message")

        try:
            # Invoke the agent with the user message
            result = await self._agent.ainvoke({"messages": [HumanMessage(content=message)]})

            # Extract the final response from the agent
            messages = result.get("messages", [])
            if messages:
                # The last message should be the AI's response
                final_message = messages[-1]
                content = (
                    final_message.content
                    if hasattr(final_message, "content")
                    else str(final_message)
                )

                # Handle case where content is a list (multimodal or tool responses)
                if isinstance(content, list):
                    # Extract text parts from the list
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    response = "\n".join(text_parts) if text_parts else str(content)
                else:
                    response = content
            else:
                response = "I apologize, but I couldn't generate a response. Please try again."

            log.info("chat_response_generated", response_length=len(response))
            return response

        except Exception as e:
            log.error(
                "chat_processing_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            # Return a user-friendly error message
            return (
                "I encountered an issue while processing your message. "
                "Please try again in a moment."
            )

    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized


# Global singleton instance
_agent_manager: AgentManager | None = None
_agent_lock = asyncio.Lock()


async def get_agent_manager(settings: Settings | None = None) -> AgentManager:
    """Get or create a shared AgentManager instance.

    This function provides lazy initialization and thread-safe access
    to the agent manager singleton.

    Args:
        settings: Application settings. If not provided, loads from environment.

    Returns:
        Initialized AgentManager instance.
    """
    global _agent_manager

    async with _agent_lock:
        if _agent_manager is None or not _agent_manager.is_initialized:
            _agent_manager = AgentManager(settings)
            await _agent_manager.initialize()
            logger.info("agent_manager_singleton_initialized")

        return _agent_manager


async def close_agent_manager() -> None:
    """Close the shared AgentManager instance.

    Call this during application shutdown to release resources.
    """
    global _agent_manager

    async with _agent_lock:
        if _agent_manager is not None:
            await _agent_manager.close()
            _agent_manager = None
            logger.info("agent_manager_singleton_closed")
