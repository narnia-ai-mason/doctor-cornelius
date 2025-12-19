"""Configuration management for Doctor Cornelius.

Uses pydantic-settings for environment variable validation and type safety.
All configuration is loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SlackSettings(BaseSettings):
    """Slack API configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SLACK_",
        extra="ignore",
    )

    bot_token: SecretStr = Field(
        ...,
        description="Slack Bot OAuth Token (xoxb-...)",
    )
    signing_secret: SecretStr = Field(
        ...,
        description="Slack Signing Secret for request verification",
    )
    app_token: SecretStr = Field(
        ...,
        description="Slack App-Level Token for Socket Mode (xapp-...)",
    )

    @field_validator("bot_token")
    @classmethod
    def validate_bot_token(cls, v: SecretStr) -> SecretStr:
        """Validate that bot token has correct prefix."""
        token = v.get_secret_value()
        if not token.startswith("xoxb-"):
            raise ValueError("Bot token must start with 'xoxb-'")
        return v

    @field_validator("app_token")
    @classmethod
    def validate_app_token(cls, v: SecretStr) -> SecretStr:
        """Validate that app token has correct prefix."""
        token = v.get_secret_value()
        if not token.startswith("xapp-"):
            raise ValueError("App token must start with 'xapp-'")
        return v


class Neo4jSettings(BaseSettings):
    """Neo4j database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="NEO4J_",
        extra="ignore",
    )

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )
    user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    password: SecretStr = Field(
        ...,
        description="Neo4j password",
    )
    max_transaction_retry_time: float = Field(
        default=30.0,
        description="Maximum time in seconds to retry failed transactions",
    )


class GeminiSettings(BaseSettings):
    """Google Gemini LLM configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    google_api_key: SecretStr = Field(
        ...,
        alias="GOOGLE_API_KEY",
        description="Google API Key for Gemini",
    )
    model: str = Field(
        default="gemini-2.0-flash",
        alias="GEMINI_MODEL",
        description="Gemini model to use for LLM operations",
    )
    embedding_model: str = Field(
        default="gemini-embedding-001",
        alias="GEMINI_EMBEDDING_MODEL",
        description="Gemini model to use for embeddings",
    )
    semaphore_limit: int = Field(
        default=10,
        alias="SEMAPHORE_LIMIT",
        description="Maximum concurrent LLM requests (for Graphiti)",
    )


class AppSettings(BaseSettings):
    """Application-level configuration."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    debug: bool = Field(
        default=False,
        alias="DEBUG",
        description="Enable debug mode",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        alias="LOG_FORMAT",
        description="Log output format: 'json' for production, 'console' for development",
    )
    app_name: str = Field(
        default="doctor-cornelius",
        description="Application name for logging and identification",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    api_host: str = Field(
        default="0.0.0.0",
        alias="API_HOST",
        description="Host to bind the API server",
    )
    api_port: int = Field(
        default=8000,
        alias="API_PORT",
        description="Port to bind the API server",
    )


class CollectorSettings(BaseSettings):
    """Data collection configuration."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    slack_rate_limit_delay: float = Field(
        default=0.5,
        alias="SLACK_RATE_LIMIT_DELAY",
        description="Delay in seconds between Slack API requests",
    )
    blocked_channel_prefixes: list[str] = Field(
        default=["external-", "guest-"],
        alias="BLOCKED_CHANNEL_PREFIXES",
        description="Channel name prefixes to exclude from collection",
    )
    batch_size: int = Field(
        default=100,
        alias="COLLECTOR_BATCH_SIZE",
        description="Number of episodes per batch during ingestion",
    )
    batch_delay_seconds: float = Field(
        default=2.0,
        alias="COLLECTOR_BATCH_DELAY",
        description="Delay between batches during ingestion",
    )
    max_concurrent_channels: int = Field(
        default=3,
        alias="MAX_CONCURRENT_CHANNELS",
        description="Maximum number of channels to process in parallel",
    )
    max_memory_queue_size: int = Field(
        default=1000,
        alias="MAX_MEMORY_QUEUE_SIZE",
        description="Maximum size of in-memory episode queue",
    )


class Settings(BaseSettings):
    """Main settings class that aggregates all configuration sections.

    Usage:
        from doctor_cornelius.config import get_settings

        settings = get_settings()
        slack_token = settings.slack.bot_token.get_secret_value()
        neo4j_uri = settings.neo4j.uri
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    slack: SlackSettings = Field(default_factory=SlackSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    collector: CollectorSettings = Field(default_factory=CollectorSettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload settings if needed.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


# Convenience function for quick access in simple scripts
def get_settings_uncached() -> Settings:
    """Get a fresh settings instance without caching.

    Useful for testing or when settings need to be reloaded.

    Returns:
        Settings: A new application settings instance.
    """
    return Settings()
