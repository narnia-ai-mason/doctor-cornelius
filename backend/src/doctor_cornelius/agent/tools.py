"""LangChain tools for the Doctor Cornelius agent.

This module defines tools that wrap Graphiti knowledge base operations
for use with LangGraph agents.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

import structlog
from langchain_core.tools import tool

if TYPE_CHECKING:
    from doctor_cornelius.knowledge.graph_client import GraphitiClientManager

logger = structlog.get_logger(__name__)

# KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))


def _utc_to_kst(date_str: str | None) -> str | None:
    """Convert UTC datetime string to KST formatted string.

    Args:
        date_str: UTC datetime string in ISO format or similar.

    Returns:
        KST formatted date string, or None if conversion fails.
    """
    if not date_str:
        return None

    try:
        # Handle various datetime string formats
        dt_str = str(date_str)

        # Try parsing ISO format with timezone
        if "T" in dt_str:
            # Remove trailing 'Z' if present and parse
            clean_str = dt_str.replace("Z", "+00:00")
            if "+" not in clean_str and "-" not in clean_str[10:]:
                # No timezone info, assume UTC
                clean_str = clean_str + "+00:00"

            try:
                dt = datetime.fromisoformat(clean_str)
            except ValueError:
                # Try parsing without microseconds
                dt = datetime.fromisoformat(clean_str.split(".")[0] + "+00:00")
        else:
            # Simple date format, just return the date part
            return dt_str.split(" ")[0]

        # Convert to KST
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        kst_dt = dt.astimezone(KST)
        return kst_dt.strftime("%Y-%m-%d %H:%M KST")

    except Exception:
        # If parsing fails, return original
        return str(date_str).split("T")[0] if date_str else None


def create_search_tool(graphiti_manager: GraphitiClientManager):
    """Create a search tool bound to a specific GraphitiClientManager instance.

    Args:
        graphiti_manager: The initialized GraphitiClientManager to use for searches.

    Returns:
        A LangChain tool function for searching the knowledge base.
    """

    @tool
    async def search_knowledge_base(query: str) -> str:
        """Search the team's knowledge base for relevant information.

        Use this tool when the user asks factual questions about:
        - Team members, projects, or decisions
        - Past conversations and discussions
        - Company policies, processes, or documentation
        - Technical details or implementations

        Do NOT use this tool for:
        - Simple greetings or casual conversation
        - General knowledge questions unrelated to the team
        - Opinions or subjective matters

        Args:
            query: The search query describing what information to find.

        Returns:
            A formatted string containing relevant facts from the knowledge base,
            or a message indicating no results were found.
        """
        log = logger.bind(query=query[:100], component="search_tool")
        log.info("searching_knowledge_base")

        try:
            results = await graphiti_manager.search(query=query, limit=10)

            if not results:
                log.info("no_results_found")
                return "No relevant information found in the knowledge base for this query."

            # Format results as a structured string for the LLM
            formatted_results: list[str] = []
            for i, result in enumerate(results, 1):
                fact = result.get("fact", "Unknown fact")
                valid_at = result.get("valid_at")

                # Build context string
                parts: list[str] = [f"{i}. {fact}"]

                # Add source/target context if available
                source_node = result.get("source_node")
                target_node = result.get("target_node")
                if source_node and source_node.get("name"):
                    context = f"({source_node['name']}"
                    if target_node and target_node.get("name"):
                        context += f" -> {target_node['name']}"
                    context += ")"
                    parts.append(context)

                # Add date in KST if available
                if valid_at:
                    kst_date = _utc_to_kst(valid_at)
                    if kst_date:
                        parts.append(f"[{kst_date}]")

                formatted_results.append(" ".join(parts))

            log.info("search_completed", result_count=len(results))
            return f"Found {len(results)} relevant facts:\n" + "\n".join(formatted_results)

        except Exception as e:
            log.error("search_failed", error_type=type(e).__name__, error=str(e))
            return f"Error searching knowledge base: {str(e)}"

    return search_knowledge_base
