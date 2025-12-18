# Doctor Cornelius - Implementation Plan

Temporal Knowledge Base AI Agent for Narnia Labs

## Project Overview

Build an extensible knowledge base system that:

1. Collects data from Slack (Phase 1 - extensible to Notion, GitHub, Jira)
2. Stores in temporal knowledge graph using Graphiti + Neo4j
3. Uses Gemini LLM for entity extraction and answer generation
4. Provides REST API and Slack DM interfaces

## Tech Stack

- **Python 3.11+** with `uv` package manager
- **slack-sdk** + **slack-bolt** for Slack integration
- **graphiti-core** for temporal knowledge graph
- **google-genai** for Gemini LLM (gemini-3-flash-preview)
- **Neo4j** for graph database
- **FastAPI** for REST API
- **APScheduler** for daily jobs

---

## Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                           DOCTOR CORNELIUS ARCHITECTURE                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-------------------+     +-------------------+     +-------------------+        |
|  |   DATA SOURCES    |     |   COLLECTORS      |     |   TRANSFORMERS    |        |
|  +-------------------+     +-------------------+     +-------------------+        |
|  |                   |     |                   |     |                   |        |
|  | [Slack] --------->|---->| SlackCollector    |---->| SlackTransformer  |        |
|  | [Notion]*-------->|---->| NotionCollector*  |---->| NotionTransformer*|        |
|  | [GitHub]*-------->|---->| GitHubCollector*  |---->| GitHubTransformer*|        |
|  | [Jira]*---------->|---->| JiraCollector*    |---->| JiraTransformer*  |        |
|  |                   |     |                   |     |                   |        |
|  | * = Future        |     | BaseCollector     |     | BaseTransformer   |        |
|  +-------------------+     +-------------------+     +-------------------+        |
|                                      |                        |                   |
|                                      v                        v                   |
|                            +-------------------+     +-------------------+        |
|                            |   SCHEDULER       |     |   EPISODE SCHEMA  |        |
|                            +-------------------+     +-------------------+        |
|                            | APScheduler       |     | Standardized      |        |
|                            | - Daily @ midnight|     | Activity/Episode  |        |
|                            | - Backfill mode   |     | Format            |        |
|                            +-------------------+     +-------------------+        |
|                                      |                        |                   |
|                                      v                        v                   |
|                            +------------------------------------------------+     |
|                            |            KNOWLEDGE GRAPH ENGINE              |     |
|                            +------------------------------------------------+     |
|                            |                                                |     |
|                            |  +------------------+   +------------------+   |     |
|                            |  | Graphiti Core    |   | Neo4j Database   |   |     |
|                            |  | - add_episode()  |   | - Temporal Store |   |     |
|                            |  | - search()       |   | - Entity Nodes   |   |     |
|                            |  | - retrieve_*()   |   | - Relation Edges |   |     |
|                            |  +------------------+   +------------------+   |     |
|                            |           |                                    |     |
|                            |           v                                    |     |
|                            |  +------------------+   +------------------+   |     |
|                            |  | Gemini LLM       |   | Gemini Embedder  |   |     |
|                            |  | - Entity Extract |   | - Vector Embed   |   |     |
|                            |  | - Relation Parse |   | - Similarity     |   |     |
|                            |  | - Answer Gen     |   | - Hybrid Search  |   |     |
|                            |  +------------------+   +------------------+   |     |
|                            |                                                |     |
|                            +------------------------------------------------+     |
|                                      |                                            |
|                                      v                                            |
|                            +------------------------------------------------+     |
|                            |              USER INTERFACES                   |     |
|                            +------------------------------------------------+     |
|                            |                                                |     |
|                            |  +------------------+   +------------------+   |     |
|                            |  | FastAPI REST     |   | Slack DM Bot     |   |     |
|                            |  | - /search        |   | - @mention       |   |     |
|                            |  | - /ingest        |   | - Direct message |   |     |
|                            |  | - /health        |   | - Thread replies |   |     |
|                            |  +------------------+   +------------------+   |     |
|                            |                                                |     |
|                            +------------------------------------------------+     |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Directory Structure

```
backend/
├── pyproject.toml
├── .env.example
├── src/
│   └── doctor_cornelius/
│       ├── __init__.py
│       ├── main.py                    # FastAPI entry point
│       ├── config.py                  # Pydantic settings
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseCollector ABC
│       │   └── slack_collector.py     # Slack implementation
│       ├── transformers/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseTransformer ABC
│       │   └── slack_transformer.py   # Slack -> Episode
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── episode.py             # Standardized Episode schema
│       ├── knowledge/
│       │   ├── __init__.py
│       │   └── graph_client.py        # Graphiti + Gemini wrapper
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   └── endpoints/
│       │       ├── search.py          # /search endpoints
│       │       ├── ingest.py          # /ingest, /backfill
│       │       └── health.py          # /health
│       ├── bot/
│       │   ├── __init__.py
│       │   └── app.py                 # Slack Bolt handlers
│       ├── scheduler/
│       │   └── jobs.py                # Daily collection job
│       └── security/
│           └── filters.py             # Channel security filters
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docker/
    └── docker-compose.yml             # Neo4j + App
```

---

## Implementation Phases

### Phase 1: Foundation & Slack Collection

**Step 1.1: Project Setup**

- Update `pyproject.toml` with all dependencies
- Create directory structure
- Create `.env.example` template
- Implement `config.py` with pydantic-settings

**Step 1.2: Base Collector Interface**

- Create `collectors/base.py` with `BaseCollector` ABC
- Define `RawDataItem` and `CollectionConfig` schemas

**Step 1.3: Slack Collector**

- Implement `SlackCollector` with:
  - `list_sources()` - list channels with security filtering
  - `collect()` - async generator for messages
  - Thread reply collection
  - Rate limiting (0.5s delay between requests)

**Step 1.4: Security Filters**

- Implement `SlackChannelFilter` in `security/filters.py`
- Exclude: archived channels, external shared channels (`is_ext_shared`, `is_shared`)
- Configurable blocked prefixes (default: `external-`, `guest-`)

### Phase 2: Knowledge Graph Integration

**Step 2.1: Episode Schema**

- Create standardized `Episode` schema in `schemas/episode.py`
- Include: name, body, source, reference_time, group_id, metadata
- Add `to_graphiti_params()` method

**Step 2.2: Slack Transformer**

- Implement `SlackTransformer` in `transformers/slack_transformer.py`
- Skip system messages (channel_join, etc.)
- Resolve user mentions
- Group thread messages as conversation episodes

**Step 2.3: Graphiti Client**

- Implement `GraphitiClientManager` in `knowledge/graph_client.py`
- Configure Gemini LLM client for entity extraction
- Configure Gemini embedder for vector search
- Methods: `ingest_episode()`, `ingest_episodes_batch()`, `search()`

### Phase 3: User Interfaces

**Step 3.1: REST API**

- `POST /search` - Search knowledge base
- `GET /search/episodes/{group_id}` - Get recent episodes
- `POST /ingest/backfill` - Trigger historical collection
- `POST /ingest/trigger-daily` - Manual daily trigger
- `GET /health` - Health check

**Step 3.2: Slack Bot (Socket Mode)**

- Implement Bolt app in `bot/app.py` with Socket Mode
- Requires `SLACK_APP_TOKEN` (xapp-...) for WebSocket connection
- `@mention` handler - Search and answer questions
- DM handler - Direct conversation with bot
- Thread reply support

**Step 3.3: Scheduler**

- APScheduler for daily midnight collection
- `run_daily_collection()` job function

---

## Key Files to Create

| File                                                     | Purpose               |
| -------------------------------------------------------- | --------------------- |
| `src/doctor_cornelius/config.py`                         | Central configuration |
| `src/doctor_cornelius/collectors/base.py`                | BaseCollector ABC     |
| `src/doctor_cornelius/collectors/slack_collector.py`     | Slack data collection |
| `src/doctor_cornelius/schemas/episode.py`                | Standardized Episode  |
| `src/doctor_cornelius/transformers/slack_transformer.py` | Slack -> Episode      |
| `src/doctor_cornelius/knowledge/graph_client.py`         | Graphiti + Gemini     |
| `src/doctor_cornelius/security/filters.py`               | Channel filtering     |
| `src/doctor_cornelius/api/router.py`                     | API routing           |
| `src/doctor_cornelius/bot/app.py`                        | Slack bot handlers    |
| `src/doctor_cornelius/main.py`                           | FastAPI app entry     |

---

## Dependencies

```toml
dependencies = [
    "fastapi>=0.125.0",
    "uvicorn>=0.38.0",
    "slack-sdk>=3.33.0",
    "slack-bolt>=1.21.0",
    "graphiti-core>=0.9.0",
    "google-genai>=1.0.0",
    "neo4j>=5.25.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "apscheduler>=3.10.0",
    "httpx>=0.28.0",
    "tenacity>=9.0.0",
    "structlog>=24.4.0",
]
```

---

## Security Guidelines

1. **Channel Filtering**: Always exclude `is_ext_shared` and `is_shared` channels
2. **Token Security**: All tokens via environment variables only
3. **Rate Limiting**: 0.5s delay between Slack API calls
4. **No DM Collection**: Only collect from channels bot is member of
5. **Audit Logging**: Log all ingestion operations

---

## Error Handling Strategy

### Neo4j: Managed Transactions (Built-in Retry)

Neo4j Python Driver automatically retries `TransientError` exceptions when using `execute_read`/`execute_write`.

```python
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError, Neo4jError

driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_user, neo4j_password),
    max_transaction_retry_time=30.0  # Retry for up to 30 seconds
)

async def safe_query(session, query, params):
    """TransientError is automatically retried"""
    try:
        return await session.execute_read(lambda tx: tx.run(query, params))
    except ServiceUnavailable:
        logger.error("neo4j_unavailable", uri=neo4j_uri)
        raise
    except Neo4jError as e:
        if e.is_retryable():
            logger.warning("neo4j_retryable_error", code=e.code)
        raise
```

### Slack API: tenacity Retry

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from slack_sdk.errors import SlackApiError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((SlackApiError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def fetch_channel_history(client, channel_id: str, oldest: str):
    return await client.conversations_history(
        channel=channel_id,
        oldest=oldest,
        limit=100
    )
```

### Graphiti/Gemini: Concurrency Control

Graphiti uses `SEMAPHORE_LIMIT` environment variable to limit concurrent LLM requests (default: 10):

```bash
# Lower if encountering 429 errors
export SEMAPHORE_LIMIT=5
```

Additionally, use tenacity for Gemini API retries:

```python
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def extract_entities(episode: Episode):
    return await graphiti_client.add_episode(...)
```

### Graceful Degradation

| Service   | Behavior on Failure                                               |
| --------- | ----------------------------------------------------------------- |
| Neo4j     | Queue episodes in memory (max 1000), flush on reconnect           |
| Gemini    | Skip entity extraction, store raw episodes for later reprocessing |
| Slack API | Skip collection cycle, retry on next schedule                     |

### Structured Error Logging

```python
import structlog

logger = structlog.get_logger()

# On error
logger.error(
    "slack_api_failed",
    channel_id=channel_id,
    error_type=type(e).__name__,
    status_code=getattr(e.response, 'status_code', None),
    retry_attempt=retry_state.attempt_number,
)
```

---

## Observability

### Structured Logging (structlog)

- JSON format (production), console format (development)
- Context: request_id, channel_id, operation_type
- Levels: DEBUG (dev), INFO (prod), ERROR (always)

### Health Checks

```python
# GET /health response
{
    "status": "healthy" | "degraded" | "unhealthy",
    "components": {
        "neo4j": {"status": "up", "latency_ms": 12},
        "slack": {"status": "up"},
        "gemini": {"status": "up"}
    },
    "version": "1.0.0"
}
```

### Metrics (Prometheus-ready for future expansion)

| Metric                    | Type      | Labels              |
| ------------------------- | --------- | ------------------- |
| `episodes_ingested_total` | Counter   | source, status      |
| `slack_api_calls_total`   | Counter   | method, status_code |
| `search_latency_seconds`  | Histogram | query_type          |

---

## Data Migration Strategy

### Schema Versioning

```python
# Store schema version in Neo4j
# (:SchemaVersion {version: "1.0.0", applied_at: datetime()})
```

### Migration Workflow

```
migrations/
├── 001_initial_schema.cypher
├── 002_add_episode_metadata.cypher
└── README.md
```

1. Check current schema version on startup
2. Apply pending migrations in order
3. Wrap each migration in a transaction
4. Record version after successful migration

### Backup

- Backup before major migrations: `neo4j-admin database dump`

---

## Backpressure Handling

### Backfill Configuration

```python
class BackfillConfig:
    batch_size: int = 100          # Episodes per batch
    batch_delay_seconds: float = 2.0  # Delay between batches
    max_concurrent_channels: int = 3  # Parallel channel processing
    max_memory_queue_size: int = 1000  # Max queue size
```

### Streaming & Memory Management

1. **Use Async Generators**: Don't load all messages at once
2. **Bounded Queue**: `asyncio.Queue(maxsize=1000)`
3. **Checkpoint**: Save progress every N episodes for failure recovery

### Graceful Shutdown

- Handle SIGTERM/SIGINT signals
- Complete current batch before shutdown
- Save checkpoint for resume

---

## Test Strategy

### Unit Tests

- `test_collectors.py` - BaseCollector, SlackCollector methods
- `test_transformers.py` - SlackTransformer, Episode conversion
- `test_security.py` - Channel filtering logic

### Integration Tests

1. **Slack API Connectivity**: `client.auth_test()`
2. **Channel List Retrieval**: `conversations_list()` with filtering
3. **Message History**: `conversations_history()` pagination
4. **Thread Replies**: `conversations_replies()` for threaded messages

### E2E Tests

- Full pipeline: Collect -> Transform -> Ingest -> Search

---

## Docker Setup

```yaml
# docker/docker-compose.yml
version: "3.8"
services:
  neo4j:
    image: neo4j:5.25-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data

  app:
    build: ..
    depends_on:
      - neo4j
    ports:
      - "8000:8000"
    env_file:
      - ../.env

volumes:
  neo4j_data:
```

---

## Environment Variables

```bash
# .env.example

# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token  # For Socket Mode

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Google Gemini
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL=gemini-3-flash-preview

# Application
DEBUG=false
LOG_LEVEL=INFO
```
