"""
Configuration management for the semantic chatbot.
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class CacheConfig:
    """Configuration for the semantic cache."""
    pinecone_api_key: str
    index_name: str = "chatbot"
    namespace: str = "default"
    similarity_threshold: float = 0.85
    ttl_days: int = 30


@dataclass
class ProviderConfig:
    """Configuration for LLM providers."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class Config:
    """Main configuration class."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        load_dotenv()

        # Cache configuration
        self.cache = CacheConfig(
            pinecone_api_key=self._get_required_env("PINECONE_API_KEY"),
            index_name=os.getenv("PINECONE_INDEX_NAME", "chatbot"),
            namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
            ttl_days=int(os.getenv("CACHE_TTL_DAYS", "30")),
        )

        # Provider configuration
        self.provider = ProviderConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_provider=os.getenv("DEFAULT_PROVIDER", "openai"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=self._get_optional_int("MAX_TOKENS"),
        )

        self._validate_config()

    @staticmethod
    def _get_required_env(key: str) -> str:
        """Get a required environment variable.

        Args:
            key: Environment variable name

        Returns:
            The environment variable value

        Raises:
            ValueError: If the environment variable is not set
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    @staticmethod
    def _get_optional_int(key: str) -> Optional[int]:
        """Get an optional integer environment variable.

        Args:
            key: Environment variable name

        Returns:
            The integer value or None if not set
        """
        value = os.getenv(key)
        return int(value) if value is not None else None

    def _validate_config(self):
        """Validate the configuration.

        Raises:
            ValueError: If the configuration is invalid
        """
        # Ensure at least one provider is configured
        if not self.provider.openai_api_key and not self.provider.anthropic_api_key:
            raise ValueError("At least one provider (OpenAI or Anthropic) must be configured")

        # Validate default provider
        if self.provider.default_provider not in ["openai", "anthropic"]:
            raise ValueError("DEFAULT_PROVIDER must be either 'openai' or 'anthropic'")

        # Validate default provider is configured
        if (
            self.provider.default_provider == "openai"
            and not self.provider.openai_api_key
        ) or (
            self.provider.default_provider == "anthropic"
            and not self.provider.anthropic_api_key
        ):
            raise ValueError(f"Default provider {self.provider.default_provider} is not configured")

        # Validate cache configuration
        if not (0 < self.cache.similarity_threshold <= 1):
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        if self.cache.ttl_days <= 0:
            raise ValueError("CACHE_TTL_DAYS must be positive")
