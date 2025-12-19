"""Search endpoints for Doctor Cornelius API.

This module provides endpoints for searching the knowledge base and
retrieving episodes from the temporal knowledge graph.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.knowledge.graph_client import GraphitiClientManager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request model for knowledge base search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    group_ids: list[str] | None = Field(
        default=None,
        description="Optional list of group IDs to restrict search scope",
    )
    center_node_uuid: str | None = Field(
        default=None,
        description="Optional UUID of a node to center the search around",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What decisions were made about the new caching layer?",
                    "limit": 10,
                    "group_ids": ["C01234567"],
                }
            ]
        }
    }


class SearchResultNode(BaseModel):
    """Node information in a search result."""

    uuid: str | None = Field(default=None, description="Node UUID")
    name: str | None = Field(default=None, description="Node name/label")


class SearchResult(BaseModel):
    """A single search result from the knowledge base."""

    fact: str = Field(description="The extracted fact or relationship")
    uuid: str = Field(description="UUID of the result edge")
    score: float | None = Field(default=None, description="Relevance score")
    source_node: SearchResultNode | None = Field(
        default=None, description="Source entity information"
    )
    target_node: SearchResultNode | None = Field(
        default=None, description="Target entity information"
    )
    created_at: str | None = Field(default=None, description="When the fact was created")
    valid_at: str | None = Field(default=None, description="When the fact was valid")
    invalid_at: str | None = Field(default=None, description="When the fact became invalid")


class SearchResponse(BaseModel):
    """Response model for knowledge base search."""

    query: str = Field(description="The original search query")
    results: list[SearchResult] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results returned")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What decisions were made about the new caching layer?",
                    "results": [
                        {
                            "fact": "Alice proposed implementing Redis for the caching layer",
                            "uuid": "abc123",
                            "score": 0.95,
                            "source_node": {"uuid": "node1", "name": "Alice"},
                            "target_node": {"uuid": "node2", "name": "Redis"},
                            "created_at": "2024-01-15T10:30:00Z",
                            "valid_at": "2024-01-15T10:30:00Z",
                        }
                    ],
                    "total_results": 1,
                }
            ]
        }
    }


class EpisodeResponse(BaseModel):
    """A single episode from the knowledge base."""

    uuid: str = Field(description="Episode UUID")
    name: str = Field(description="Episode name/title")
    content: str = Field(description="Episode content")
    source: str | None = Field(default=None, description="Episode source type")
    source_description: str | None = Field(default=None, description="Description of the source")
    created_at: str | None = Field(default=None, description="When created")
    valid_at: str | None = Field(default=None, description="When valid")
    group_id: str = Field(description="Episode group ID")


class EpisodesListResponse(BaseModel):
    """Response model for episode retrieval."""

    group_id: str = Field(description="The group ID that was queried")
    episodes: list[EpisodeResponse] = Field(description="List of episodes")
    total_episodes: int = Field(description="Total number of episodes returned")


# -----------------------------------------------------------------------------
# Dependency Injection
# -----------------------------------------------------------------------------


async def get_graph_client(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> GraphitiClientManager:
    """Dependency to get an initialized GraphitiClientManager.

    This creates a new client instance for each request. For production,
    consider using a connection pool or application-level singleton.

    Args:
        settings: Application settings.

    Returns:
        An initialized GraphitiClientManager.

    Raises:
        HTTPException: If client initialization fails.
    """
    client = GraphitiClientManager(settings=settings)
    try:
        await client.initialize()
        return client
    except Exception as e:
        logger.error(
            "graph_client_initialization_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Knowledge graph service unavailable: {str(e)}",
        ) from e


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search Knowledge Base",
    description="Search the temporal knowledge graph for relevant facts and relationships.",
    responses={
        200: {"description": "Search completed successfully"},
        503: {"description": "Knowledge graph service unavailable"},
    },
)
async def search_knowledge_base(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008 - Dependency injection
) -> SearchResponse:
    """Search the knowledge base with a natural language query.

    This endpoint performs a hybrid search combining vector similarity and
    graph traversal to find the most relevant facts and entities.

    Args:
        request: The search request containing query and filters.
        settings: Application settings.

    Returns:
        SearchResponse containing matching results.

    Raises:
        HTTPException: If search fails or service is unavailable.
    """
    logger.info(
        "search_request_received",
        query=request.query[:100],
        limit=request.limit,
        group_ids=request.group_ids,
    )

    client = GraphitiClientManager(settings=settings)
    try:
        await client.initialize()

        results = await client.search(
            query=request.query,
            limit=request.limit,
            group_ids=request.group_ids,
            center_node_uuid=request.center_node_uuid,
        )

        # Transform results to response model
        search_results = []
        for result in results:
            source_node = None
            if result.get("source_node"):
                source_node = SearchResultNode(
                    uuid=result["source_node"].get("uuid"),
                    name=result["source_node"].get("name"),
                )

            target_node = None
            if result.get("target_node"):
                target_node = SearchResultNode(
                    uuid=result["target_node"].get("uuid"),
                    name=result["target_node"].get("name"),
                )

            search_results.append(
                SearchResult(
                    fact=result.get("fact", ""),
                    uuid=result.get("uuid", ""),
                    score=result.get("score"),
                    source_node=source_node,
                    target_node=target_node,
                    created_at=result.get("created_at"),
                    valid_at=result.get("valid_at"),
                    invalid_at=result.get("invalid_at"),
                )
            )

        logger.info(
            "search_completed",
            query=request.query[:50],
            result_count=len(search_results),
        )

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
        )

    except Exception as e:
        logger.error(
            "search_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            query=request.query[:50],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        ) from e
    finally:
        await client.close()


@router.get(
    "/episodes/{group_id}",
    response_model=EpisodesListResponse,
    summary="Get Recent Episodes",
    description="Retrieve recent episodes for a specific group (e.g., Slack channel).",
    responses={
        200: {"description": "Episodes retrieved successfully"},
        404: {"description": "Group not found or no episodes"},
        503: {"description": "Knowledge graph service unavailable"},
    },
)
async def get_episodes_by_group(
    group_id: str,
    limit: int = Query(
        default=10, ge=1, le=100, description="Maximum number of episodes to return"
    ),
    reference_time: datetime | None = Query(  # noqa: B008 - Dependency injection
        default=None,
        description="Reference time for temporal filtering (ISO 8601 format)",
    ),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> EpisodesListResponse:
    """Retrieve recent episodes for a specific group.

    This endpoint fetches historical episodes from the knowledge graph
    for a given group ID (e.g., Slack channel ID).

    Args:
        group_id: The group ID to retrieve episodes for.
        limit: Maximum number of episodes to return.
        reference_time: Optional reference time for temporal filtering.
        settings: Application settings.

    Returns:
        EpisodesListResponse containing the episodes.

    Raises:
        HTTPException: If retrieval fails or service is unavailable.
    """
    logger.info(
        "episodes_request_received",
        group_id=group_id,
        limit=limit,
        reference_time=reference_time,
    )

    client = GraphitiClientManager(settings=settings)
    try:
        await client.initialize()

        # Use provided reference time or current UTC
        ref_time = reference_time or datetime.now(UTC)

        episodes = await client.retrieve_episodes(
            group_id=group_id,
            limit=limit,
            reference_time=ref_time,
        )

        # Transform results to response model
        episode_responses = [
            EpisodeResponse(
                uuid=ep.get("uuid", ""),
                name=ep.get("name", ""),
                content=ep.get("content", ""),
                source=ep.get("source"),
                source_description=ep.get("source_description"),
                created_at=ep.get("created_at"),
                valid_at=ep.get("valid_at"),
                group_id=ep.get("group_id", group_id),
            )
            for ep in episodes
        ]

        logger.info(
            "episodes_retrieved",
            group_id=group_id,
            episode_count=len(episode_responses),
        )

        return EpisodesListResponse(
            group_id=group_id,
            episodes=episode_responses,
            total_episodes=len(episode_responses),
        )

    except Exception as e:
        logger.error(
            "episodes_retrieval_failed",
            error_type=type(e).__name__,
            error_message=str(e),
            group_id=group_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Episode retrieval failed: {str(e)}",
        ) from e
    finally:
        await client.close()


# Export public API
__all__ = [
    "router",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchResultNode",
    "EpisodeResponse",
    "EpisodesListResponse",
]
