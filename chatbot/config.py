"""
Configuration management for the semantic chatbot.
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dotenv import load_dotenv


class VectorStoreType(Enum):
    """Supported vector store types."""
    PINECONE = "pinecone"
    QDRANT = "qdrant"


@dataclass
class CacheConfig:
    """Configuration for the semantic cache."""
    vector_store: VectorStoreType
    # Pinecone settings
    pinecone_api_key: Optional[str] = None
    # Qdrant settings
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    # Common settings
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

    def __init__(
        self,
        vector_store_override: Optional[str] = None,
        ttl_days_override: Optional[int] = None
    ):
        """Initialize configuration from environment variables.
        
        Args:
            vector_store_override: Optional vector store type to use instead of environment variable
            ttl_days_override: Optional TTL days to use instead of environment variable
        """
        load_dotenv()

        # Cache configuration
        vector_store = vector_store_override or os.getenv("VECTOR_STORE", "pinecone").lower()
        try:
            vector_store_type = VectorStoreType(vector_store)
        except ValueError:
            raise ValueError(f"Invalid vector store type: {vector_store}")

        self.cache = CacheConfig(
            vector_store=vector_store_type,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            index_name=os.getenv("VECTOR_INDEX_NAME", "chatbot"),
            namespace=os.getenv("VECTOR_NAMESPACE", "default"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.85")),
            ttl_days=ttl_days_override if ttl_days_override is not None else int(os.getenv("CACHE_TTL_DAYS", "30")),
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
        # Validate vector store configuration
        if self.cache.vector_store == VectorStoreType.PINECONE:
            if not self.cache.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY is required when using Pinecone")
        elif self.cache.vector_store == VectorStoreType.QDRANT:
            if not self.cache.qdrant_url:
                raise ValueError("QDRANT_URL is required when using Qdrant")

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
        if self.cache.ttl_days < 0:
            raise ValueError("CACHE_TTL_DAYS must be non-negative")
