"""
Tests for the cache manager.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chatbot.cache.manager import CacheManager
from chatbot.cache.vector_store import CacheEntry, VectorStore
from chatbot.config import Config, CacheConfig, ProviderConfig
from chatbot.providers.base import LLMProvider


@pytest.fixture
def vector_store():
    """Create a mock vector store."""
    with patch('chatbot.cache.manager.VectorStore', autospec=True) as mock:
        instance = mock.return_value
        instance.find_similar = AsyncMock()
        instance.store = AsyncMock()
        instance.cleanup_old_entries = MagicMock()
        yield instance


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
    mock_config = MagicMock(spec=Config)
    
    # Mock cache config
    mock_config.cache = MagicMock(spec=CacheConfig)
    mock_config.cache.pinecone_api_key = "test_api_key"
    mock_config.cache.index_name = "test_index"
    mock_config.cache.namespace = "test_namespace"
    mock_config.cache.similarity_threshold = 0.85
    mock_config.cache.ttl_days = 30
    
    # Mock provider config
    mock_config.provider = MagicMock()
    mock_config.provider.temperature = 0.7
    mock_config.provider.max_tokens = None
    
    return mock_config


@pytest.fixture
def openai_client():
    """Create a mock OpenAI client."""
    return AsyncMock()


@pytest.fixture
def provider():
    """Create a mock LLM provider."""
    provider = MockProvider("fake_api_key")
    provider.generate_response = AsyncMock()
    return provider


@pytest.fixture
def cache_manager(config, openai_client, provider, vector_store):
    """Create a CacheManager instance with mocked dependencies."""
    with patch('chatbot.cache.manager.VectorStore', return_value=vector_store):
        manager = CacheManager(config, openai_client, provider)
        return manager


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
    response, from_cache, cache_info = await cache_manager.get_response("test question")

    # Verify cache was used
    assert response == "cached answer"
    assert from_cache is True
    assert cache_info == ("test question", 0.95)
    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_not_called()
    cache_manager.provider.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_get_response_cache_miss(cache_manager):
    """Test getting a response when cache misses."""
    # Mock cache miss
    cache_manager.vector_store.find_similar = AsyncMock(return_value=[])
    cache_manager.vector_store.store = AsyncMock()
    cache_manager.provider.generate_response.return_value = "Generated response"

    # Get response
    response, from_cache, cache_info = await cache_manager.get_response("test question")

    # Verify LLM was used and response was cached
    assert response == "Generated response"
    assert from_cache is False
    assert cache_info is None
    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_called_once()
    cache_manager.provider.generate_response.assert_called_once_with(
        prompt="test question",
        system_prompt=None,
        temperature=0.7,
        max_tokens=None
    )


@pytest.mark.asyncio
async def test_get_response_with_system_prompt(cache_manager):
    """Test getting a response with a custom system prompt."""
    # Mock cache miss
    cache_manager.vector_store.find_similar = AsyncMock(return_value=[])
    cache_manager.vector_store.store = AsyncMock()
    cache_manager.provider.generate_response.return_value = "Generated response with system prompt"

    # Get response with system prompt
    system_prompt = "Custom system prompt"
    response, from_cache, cache_info = await cache_manager.get_response("test question", system_prompt=system_prompt)

    # Verify system prompt was passed to provider
    assert response == "Generated response with system prompt"
    assert from_cache is False
    assert cache_info is None
    cache_manager.provider.generate_response.assert_called_once_with(
        prompt="test question",
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=None
    )


@pytest.mark.asyncio
async def test_cleanup(cache_manager):
    """Test cache cleanup."""
    # Mock cleanup to return number of entries removed
    expected_removed = 5
    cache_manager.vector_store.cleanup_old_entries = MagicMock(return_value=expected_removed)

    # Run cleanup
    removed = await cache_manager.cleanup()

    # Verify cleanup was called and returned correct number
    assert removed == expected_removed
    cache_manager.vector_store.cleanup_old_entries.assert_called_once_with()
    
    # Verify no other vector store methods were called
    cache_manager.vector_store.find_similar.assert_not_called()
    cache_manager.vector_store.store.assert_not_called()
