"""LangGraph agent definition for Doctor Cornelius.

This module defines the ReAct agent that powers Doctor Cornelius,
using LangGraph and Google Gemini for intelligent conversation
with knowledge base integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from doctor_cornelius.agent.tools import create_search_tool

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

    from doctor_cornelius.config import Settings
    from doctor_cornelius.knowledge.graph_client import GraphitiClientManager

logger = structlog.get_logger(__name__)

# System prompt defining Doctor Cornelius's personality and behavior
SYSTEM_PROMPT = """You are Doctor Cornelius, the trusted archivist of Narnia Labs.
You help team members find information from past conversations, documents, and team knowledge.

## Your Personality
- Friendly, helpful, and professional
- Knowledgeable but humble - you acknowledge when you don't know something
- Concise in your responses - get to the point without unnecessary verbosity

## Time Zone
- The team is based in Korea (KST, UTC+9)
- All times in search results are already converted to KST
- When mentioning dates/times, use KST format

## Guidelines

### When to Search the Knowledge Base
Use the search_knowledge_base tool when users ask about:
- Team members, their roles, or expertise
- Past projects, decisions, or discussions
- Company processes, policies, or documentation
- Technical implementations or architecture details
- Meeting notes or action items

### When NOT to Search
Respond directly without searching for:
- Simple greetings (hi, hello, thanks, etc.)
- General knowledge questions unrelated to the team
- Casual conversation or small talk
- Questions you can answer from the conversation context

### Response Style
- Be natural and conversational, not robotic
- When sharing information from the knowledge base, integrate it naturally into your response
- If you found relevant information, mention the date/timeframe when appropriate
- If no relevant information was found, be honest about it and offer to help differently
- Keep responses focused and actionable
"""


def create_doctor_cornelius_agent(
    settings: Settings,
    graphiti_manager: GraphitiClientManager,
) -> CompiledGraph:
    """Create the Doctor Cornelius LangGraph agent.

    Args:
        settings: Application settings containing API keys and model configuration.
        graphiti_manager: Initialized GraphitiClientManager for knowledge base operations.

    Returns:
        A compiled LangGraph agent ready to process messages.
    """
    log = logger.bind(
        component="agent_factory",
        model=settings.gemini.model,
    )
    log.info("creating_doctor_cornelius_agent")

    # Initialize the Gemini model for LangChain
    model = ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        google_api_key=settings.gemini.google_api_key.get_secret_value(),
        temperature=0.7,
        convert_system_message_to_human=True,
    )

    # Create tools bound to the Graphiti manager
    search_tool = create_search_tool(graphiti_manager)
    tools = [search_tool]

    # Create the ReAct agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    log.info("agent_created")
    return agent
