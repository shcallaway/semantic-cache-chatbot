"""
Tests for the cache manager.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chatbot.cache.manager import CacheManager
from chatbot.cache.vector_store import CacheEntry
from chatbot.config import Config
from chatbot.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    async def generate_response(self, prompt, system_prompt=None, temperature=0.7, max_tokens=None):
        return f"Response to: {prompt}"

    def get_token_count(self, text):
        return len(text) // 4

    @property
    def provider_name(self):
        return "mock"

    @property
    def default_system_prompt(self):
        return "Mock system prompt"


@pytest.fixture
def config():
    """Create a test configuration."""
    return MagicMock(spec=Config)


@pytest.fixture
def openai_client():
    """Create a mock OpenAI client."""
    return AsyncMock()


@pytest.fixture
def provider():
    """Create a mock LLM provider."""
    return MockProvider("fake_api_key")


@pytest.fixture
def cache_manager(config, openai_client, provider):
    """Create a CacheManager instance with mocked dependencies."""
    return CacheManager(config, openai_client, provider)


@pytest.mark.asyncio
async def test_get_response_cache_hit(cache_manager):
    """Test getting a response from cache."""
    # Mock cache hit
    cache_entry = CacheEntry(
        question="test question",
        answer="cached answer",
        provider="mock",
        timestamp=1234567890,
        embedding=[0.1] * 1536
    )
    
    cache_manager.vector_store.find_similar = AsyncMock(
        return_value=[(cache_entry, 0.95)]
    )

    # Get response
    response, from_cache = await cache_manager.get_response("test question")

    # Verify cache was used
    assert response == "cached answer"
    assert from_cache is True
    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_not_called()


@pytest.mark.asyncio
async def test_get_response_cache_miss(cache_manager):
    """Test getting a response when cache misses."""
    # Mock cache miss
    cache_manager.vector_store.find_similar = AsyncMock(return_value=[])
    cache_manager.vector_store.store = AsyncMock()

    # Get response
    response, from_cache = await cache_manager.get_response("test question")

    # Verify LLM was used and response was cached
    assert response == "Response to: test question"
    assert from_cache is False
    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_called_once()


@pytest.mark.asyncio
async def test_get_response_with_system_prompt(cache_manager):
    """Test getting a response with a custom system prompt."""
    # Mock cache miss
    cache_manager.vector_store.find_similar = AsyncMock(return_value=[])
    cache_manager.vector_store.store = AsyncMock()

    # Get response with system prompt
    system_prompt = "Custom system prompt"
    await cache_manager.get_response("test question", system_prompt=system_prompt)

    # Verify system prompt was passed to provider
    assert cache_manager.provider._system_prompt != system_prompt


@pytest.mark.asyncio
async def test_cleanup(cache_manager):
    """Test cache cleanup."""
    # Mock cleanup
    cache_manager.vector_store.cleanup_old_entries = MagicMock(return_value=5)

    # Run cleanup
    removed = await cache_manager.cleanup()

    # Verify cleanup was called
    assert removed == 5
    cache_manager.vector_store.cleanup_old_entries.assert_called_once()
