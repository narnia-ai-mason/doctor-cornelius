"""Graphiti client wrapper for knowledge graph operations.

This module provides a managed Graphiti client with Gemini LLM and embedder
integration for the Doctor Cornelius temporal knowledge base.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from neo4j.exceptions import ClientError as Neo4jClientError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    pass

from doctor_cornelius.config import Settings, get_settings
from doctor_cornelius.schemas.episode import Episode

logger = structlog.get_logger(__name__)


def _is_retryable_gemini_error(exception: BaseException) -> bool:
    """Check if an exception is a retryable Gemini API error.

    Handles rate limit (429) and service unavailable (503) errors from
    both google-genai SDK (APIError) and google-api-core (ResourceExhausted).

    Args:
        exception: The exception to check.

    Returns:
        True if the exception indicates a retryable error.
    """
    # Check for google.genai.errors.APIError (newer SDK)
    try:
        from google.genai.errors import APIError

        if isinstance(exception, APIError):
            # 429 = rate limit, 503 = service unavailable
            return exception.code in (429, 503)
    except ImportError:
        pass

    # Check for google.api_core.exceptions (older/transitive dependency)
    try:
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

        if isinstance(exception, (ResourceExhausted, ServiceUnavailable)):
            return True
    except ImportError:
        pass

    # Check for generic HTTP errors with 429/503 status codes
    error_str = str(exception).lower()
    if "429" in error_str or "resource exhausted" in error_str:
        return True

    return "503" in error_str or "service unavailable" in error_str


class GraphitiClientManager:
    """Managed Graphiti client with Gemini LLM and embedder integration.

    This class provides a wrapper around the Graphiti client with:
    - Gemini LLM for entity extraction
    - Gemini embedder for vector search
    - Automatic retry handling for API rate limits
    - Context manager support for proper resource cleanup

    Usage:
        async with GraphitiClientManager() as client:
            await client.ingest_episode(episode)
            results = await client.search("query")

    Or manually:
        manager = GraphitiClientManager()
        await manager.initialize()
        try:
            await manager.ingest_episode(episode)
        finally:
            await manager.close()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the GraphitiClientManager.

        Args:
            settings: Application settings. If not provided, loads from environment.
        """
        self._settings = settings or get_settings()
        self._client: Graphiti | None = None
        self._initialized = False

        # Set SEMAPHORE_LIMIT environment variable for Graphiti's internal concurrency
        os.environ["SEMAPHORE_LIMIT"] = str(self._settings.gemini.semaphore_limit)

        self._log = logger.bind(
            component="graphiti_client",
            neo4j_uri=self._settings.neo4j.uri,
            gemini_model=self._settings.gemini.model,
            embedding_model=self._settings.gemini.embedding_model,
        )

    async def initialize(self) -> None:
        """Initialize the Graphiti client with Gemini LLM and embedder.

        This sets up:
        - Gemini LLM client for entity extraction
        - Gemini embedder for vector search
        - Gemini reranker for cross-encoder functionality
        - Neo4j database connection and indices

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            self._log.debug("client_already_initialized")
            return

        self._log.info("initializing_graphiti_client")

        try:
            api_key = self._settings.gemini.google_api_key.get_secret_value()

            # Configure Gemini LLM client
            llm_config = LLMConfig(
                api_key=api_key,
                model=self._settings.gemini.model,
            )
            llm_client = GeminiClient(config=llm_config)

            # Configure Gemini embedder
            embedder_config = GeminiEmbedderConfig(
                api_key=api_key,
                embedding_model=self._settings.gemini.embedding_model,
            )
            embedder = GeminiEmbedder(config=embedder_config)

            # Configure Gemini reranker (cross-encoder)
            reranker_config = LLMConfig(
                api_key=api_key,
                model=self._settings.gemini.model,
            )
            cross_encoder = GeminiRerankerClient(config=reranker_config)

            # Initialize Graphiti client
            self._client = Graphiti(
                self._settings.neo4j.uri,
                self._settings.neo4j.user,
                self._settings.neo4j.password.get_secret_value(),
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )

            # Build indices and constraints
            # Note: Neo4j 5.x throws EquivalentSchemaRuleAlreadyExists even with IF NOT EXISTS
            # when an equivalent index already exists. We catch and ignore this error.
            try:
                await self._client.build_indices_and_constraints()
            except Neo4jClientError as e:
                if "EquivalentSchemaRuleAlreadyExists" in str(e):
                    self._log.debug("neo4j_indices_already_exist", message=str(e))
                else:
                    raise

            self._initialized = True
            self._log.info("graphiti_client_initialized")

        except Exception as e:
            self._log.error(
                "graphiti_initialization_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise RuntimeError(f"Failed to initialize Graphiti client: {e}") from e

    async def close(self) -> None:
        """Close the Graphiti client and release resources."""
        if self._client is not None:
            self._log.info("closing_graphiti_client")
            try:
                await self._client.close()
            except Exception as e:
                self._log.warning(
                    "graphiti_close_warning",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            finally:
                self._client = None
                self._initialized = False
                self._log.info("graphiti_client_closed")

    async def __aenter__(self) -> GraphitiClientManager:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> Graphiti:
        """Ensure the client is initialized and return it.

        Returns:
            The initialized Graphiti client.

        Raises:
            RuntimeError: If the client is not initialized.
        """
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "GraphitiClientManager is not initialized. "
                "Call initialize() or use as context manager."
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable_gemini_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def ingest_episode(self, episode: Episode) -> dict[str, Any]:
        """Add a single episode to the knowledge graph.

        This method extracts entities and relationships from the episode
        content using Gemini LLM and stores them in the Neo4j graph database.

        Args:
            episode: The episode to ingest.

        Returns:
            A dictionary containing:
                - episode_uuid: UUID of the created episode node
                - entities: List of extracted entity UUIDs
                - relationships: List of extracted relationship UUIDs

        Raises:
            RuntimeError: If the client is not initialized.
            ResourceExhausted: If Gemini API rate limit is exceeded (after retries).
            ServiceUnavailable: If Gemini service is unavailable (after retries).
        """
        client = self._ensure_initialized()

        log = self._log.bind(
            episode_name=episode.name,
            episode_source=episode.source.value,
            group_id=episode.group_id,
        )
        log.debug("ingesting_episode")

        try:
            params = episode.to_graphiti_params()

            result = await client.add_episode(
                name=params["name"],
                episode_body=params["episode_body"],
                source=EpisodeType(params["source"]),
                source_description=params["source_description"],
                reference_time=params["reference_time"],
                group_id=params["group_id"],
            )

            response = {
                "episode_uuid": result.episode.uuid if result.episode else None,
                "entities": [node.uuid for node in result.nodes] if result.nodes else [],
                "relationships": [edge.uuid for edge in result.edges] if result.edges else [],
            }

            log.info(
                "episode_ingested",
                episode_uuid=response["episode_uuid"],
                entity_count=len(response["entities"]),
                relationship_count=len(response["relationships"]),
            )

            return response

        except Exception as e:
            if _is_retryable_gemini_error(e):
                log.warning("gemini_api_rate_limited", episode_name=episode.name)
                raise
            log.error(
                "episode_ingestion_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable_gemini_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def ingest_episodes_batch(
        self,
        episodes: list[Episode],
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """Add multiple episodes to the knowledge graph in batch.

        This method uses Graphiti's bulk ingestion API for efficient
        processing of multiple episodes. All episodes are processed
        as a single batch with the same group_id.

        Args:
            episodes: List of episodes to ingest.
            group_id: Optional group ID override. If not provided, uses
                the group_id from the first episode.

        Returns:
            A dictionary containing:
                - episode_count: Number of episodes ingested
                - entities: List of all extracted entity UUIDs
                - relationships: List of all extracted relationship UUIDs

        Raises:
            RuntimeError: If the client is not initialized.
            ValueError: If episodes list is empty.
            ResourceExhausted: If Gemini API rate limit is exceeded (after retries).
            ServiceUnavailable: If Gemini service is unavailable (after retries).
        """
        if not episodes:
            raise ValueError("Episodes list cannot be empty")

        client = self._ensure_initialized()

        # Use provided group_id or get from first episode
        effective_group_id = group_id or episodes[0].group_id

        log = self._log.bind(
            episode_count=len(episodes),
            group_id=effective_group_id,
        )
        log.info("ingesting_episodes_batch")

        try:
            # Convert episodes to RawEpisode format
            raw_episodes = []
            for episode in episodes:
                params = episode.to_raw_episode_params()
                raw_episodes.append(
                    RawEpisode(
                        name=params["name"],
                        content=params["content"],
                        source=EpisodeType(params["source"]),
                        source_description=params["source_description"],
                        reference_time=params["reference_time"],
                    )
                )

            result = await client.add_episode_bulk(
                bulk_episodes=raw_episodes,
                group_id=effective_group_id,
            )

            response = {
                "episode_count": len(result.episodes) if result.episodes else 0,
                "entities": [node.uuid for node in result.nodes] if result.nodes else [],
                "relationships": [edge.uuid for edge in result.edges] if result.edges else [],
            }

            log.info(
                "episodes_batch_ingested",
                episode_count=response["episode_count"],
                entity_count=len(response["entities"]),
                relationship_count=len(response["relationships"]),
            )

            return response

        except Exception as e:
            if _is_retryable_gemini_error(e):
                log.warning("gemini_api_rate_limited_batch", episode_count=len(episodes))
                raise
            log.error(
                "batch_ingestion_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable_gemini_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        limit: int = 10,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base for relevant information.

        Performs a hybrid search combining vector similarity and graph
        traversal to find the most relevant facts and entities.

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return (default: 10).
            group_ids: Optional list of group IDs to restrict search scope.
            center_node_uuid: Optional UUID of a node to center the search around.

        Returns:
            A list of search result dictionaries, each containing:
                - fact: The extracted fact or relationship
                - uuid: UUID of the result edge
                - score: Relevance score
                - source_node: Source entity information
                - target_node: Target entity information
                - created_at: When the fact was created
                - valid_at: When the fact was valid

        Raises:
            RuntimeError: If the client is not initialized.
            ResourceExhausted: If Gemini API rate limit is exceeded (after retries).
            ServiceUnavailable: If Gemini service is unavailable (after retries).
        """
        client = self._ensure_initialized()

        log = self._log.bind(
            query=query[:100],  # Truncate for logging
            limit=limit,
            group_ids=group_ids,
            center_node_uuid=center_node_uuid,
        )
        log.debug("searching_knowledge_base")

        try:
            results = await client.search(
                query=query,
                num_results=limit,
                center_node_uuid=center_node_uuid,
                group_ids=group_ids,
            )

            # Transform results to serializable format
            search_results = []
            for result in results:
                search_results.append(
                    {
                        "fact": result.fact,
                        "uuid": result.uuid,
                        "score": getattr(result, "score", None),
                        "source_node": {
                            "uuid": result.source_node_uuid,
                            "name": getattr(result, "source_node_name", None),
                        }
                        if hasattr(result, "source_node_uuid")
                        else None,
                        "target_node": {
                            "uuid": result.target_node_uuid,
                            "name": getattr(result, "target_node_name", None),
                        }
                        if hasattr(result, "target_node_uuid")
                        else None,
                        "created_at": str(result.created_at)
                        if hasattr(result, "created_at")
                        else None,
                        "valid_at": str(result.valid_at) if hasattr(result, "valid_at") else None,
                        "invalid_at": (
                            str(result.invalid_at)
                            if hasattr(result, "invalid_at") and result.invalid_at
                            else None
                        ),
                    }
                )

            log.info(
                "search_completed",
                result_count=len(search_results),
            )

            return search_results

        except Exception as e:
            if _is_retryable_gemini_error(e):
                log.warning("gemini_api_rate_limited_search", query=query[:50])
                raise
            log.error(
                "search_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    async def retrieve_episodes(
        self,
        group_id: str,
        limit: int = 10,
        reference_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve recent episodes for a specific group.

        Fetches historical episodes from the knowledge graph, optionally
        filtered by reference time.

        Args:
            group_id: The group ID to retrieve episodes for.
            limit: Maximum number of episodes to return (default: 10).
            reference_time: Optional reference time for temporal filtering.
                If not provided, uses current UTC time.

        Returns:
            A list of episode dictionaries, each containing:
                - uuid: Episode UUID
                - name: Episode name
                - content: Episode content
                - source: Episode source type
                - source_description: Description of the source
                - created_at: When the episode was created
                - valid_at: When the episode was valid
                - group_id: The episode's group ID

        Raises:
            RuntimeError: If the client is not initialized.
        """
        client = self._ensure_initialized()

        log = self._log.bind(
            group_id=group_id,
            limit=limit,
        )
        log.debug("retrieving_episodes")

        try:
            ref_time = reference_time or datetime.now(UTC)

            episodes = await client.retrieve_episodes(
                reference_time=ref_time,
                last_n=limit,
                group_ids=[group_id],
            )

            # Transform episodes to serializable format
            episode_list = []
            for ep in episodes:
                episode_list.append(
                    {
                        "uuid": ep.uuid,
                        "name": ep.name,
                        "content": ep.content,
                        "source": ep.source.value if ep.source else None,
                        "source_description": getattr(ep, "source_description", None),
                        "created_at": str(ep.created_at) if ep.created_at else None,
                        "valid_at": str(ep.valid_at) if ep.valid_at else None,
                        "group_id": ep.group_id,
                    }
                )

            log.info(
                "episodes_retrieved",
                episode_count=len(episode_list),
            )

            return episode_list

        except Exception as e:
            log.error(
                "episode_retrieval_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._initialized

    @property
    def client(self) -> Graphiti | None:
        """Access the underlying Graphiti client.

        Returns None if not initialized.
        """
        return self._client
