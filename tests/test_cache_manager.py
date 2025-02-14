"""
Tests for the cache manager.

This module contains tests for the CacheManager class, including:
- Basic functionality (cache hits/misses)
- Vector store creation and configuration
- Error handling
- Edge cases and input validation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chatbot.cache.base import CacheEntry
from chatbot.cache.manager import CacheManager
from chatbot.config import Config, CacheConfig, VectorStoreType
from chatbot.providers.base import LLMProvider


@pytest.fixture(params=[VectorStoreType.PINECONE, VectorStoreType.QDRANT, VectorStoreType.PGVECTOR])
def vector_store_type(request):
    """Parameterized fixture for testing both vector store types."""
    return request.param


@pytest.fixture
def vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock.find_similar = AsyncMock()
    mock.store = AsyncMock()
    mock.cleanup_old_entries = AsyncMock()
    return mock


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    async def generate_response(
        self, prompt, system_prompt=None, temperature=0.7, max_tokens=None
    ):
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
def config(vector_store_type):
    """Create a test configuration."""
    mock_config = MagicMock(spec=Config)

    # Mock cache config
    mock_config.cache = MagicMock(spec=CacheConfig)
    mock_config.cache.vector_store = vector_store_type
    mock_config.cache.pinecone_api_key = "test_pinecone_key"
    mock_config.cache.qdrant_url = "http://test:6333"
    mock_config.cache.qdrant_api_key = "test_qdrant_key"
    mock_config.cache.postgres_host = "localhost"
    mock_config.cache.postgres_port = 5432
    mock_config.cache.postgres_user = "test_user"
    mock_config.cache.postgres_password = "test_password"
    mock_config.cache.postgres_db = "test_db"
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
def cache_manager_no_patches(config, openai_client, provider):
    """Create a CacheManager instance without patching vector stores."""
    return CacheManager(config, openai_client, provider)


@pytest.fixture
def cache_manager(config, openai_client, provider, vector_store):
    """Create a CacheManager instance with mocked dependencies."""
    # Patch both vector store implementations at the module level
    patches = [
        patch("chatbot.cache.manager.PineconeVectorStore", return_value=vector_store),
        patch("chatbot.cache.manager.QdrantVectorStore", return_value=vector_store),
        patch("chatbot.cache.manager.PgVectorStore", return_value=vector_store),
    ]

    for p in patches:
        p.start()

    manager = CacheManager(config, openai_client, provider)

    # Stop all patches after creating the manager
    for p in patches:
        p.stop()

    return manager


def test_create_vector_store_invalid_type(config, openai_client):
    """Test creating vector store with invalid type raises error."""
    config.cache.vector_store = "invalid"
    with pytest.raises(ValueError, match="Invalid vector store type: invalid"):
        CacheManager(config, openai_client, MockProvider("fake_key"))


def test_create_pinecone_store(config, openai_client):
    """Test Pinecone store is created with correct config."""
    config.cache.vector_store = VectorStoreType.PINECONE
    with patch("chatbot.cache.manager.PineconeVectorStore") as mock_pinecone:
        CacheManager(config, openai_client, MockProvider("fake_key"))
        mock_pinecone.assert_called_once_with(config.cache, openai_client)


def test_create_qdrant_store(config, openai_client):
    """Test Qdrant store is created with correct config."""
    config.cache.vector_store = VectorStoreType.QDRANT
    with patch("chatbot.cache.manager.QdrantVectorStore") as mock_qdrant:
        CacheManager(config, openai_client, MockProvider("fake_key"))
        mock_qdrant.assert_called_once_with(config.cache, openai_client)


def test_create_pgvector_store(config, openai_client):
    """Test pgvector store is created with correct config."""
    config.cache.vector_store = VectorStoreType.PGVECTOR
    with patch("chatbot.cache.manager.PgVectorStore") as mock_pgvector:
        CacheManager(config, openai_client, MockProvider("fake_key"))
        mock_pgvector.assert_called_once_with(config.cache, openai_client)


@pytest.mark.asyncio
async def test_get_response_store_error(cache_manager):
    """Test handling of vector store errors."""
    cache_manager.vector_store.find_similar.side_effect = Exception("Store error")

    with pytest.raises(Exception, match="Store error"):
        await cache_manager.get_response("test question")

    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_not_called()
    cache_manager.provider.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_get_response_llm_error(cache_manager):
    """Test handling of LLM provider errors."""
    cache_manager.vector_store.find_similar.return_value = []
    cache_manager.provider.generate_response.side_effect = Exception("LLM error")

    with pytest.raises(Exception, match="LLM error"):
        await cache_manager.get_response("test question")

    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_not_called()


@pytest.mark.asyncio
async def test_get_response_store_error_on_cache(cache_manager):
    """Test handling of vector store errors during caching."""
    cache_manager.vector_store.find_similar.return_value = []
    cache_manager.provider.generate_response.return_value = "Generated response"
    cache_manager.vector_store.store.side_effect = Exception("Store error")

    with pytest.raises(Exception, match="Store error"):
        await cache_manager.get_response("test question")

    cache_manager.vector_store.find_similar.assert_called_once()
    cache_manager.vector_store.store.assert_called_once()
    cache_manager.provider.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_get_response_empty_question(cache_manager):
    """Test handling of empty/whitespace questions."""
    for empty_input in ["", " ", "\n", "\t"]:
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await cache_manager.get_response(empty_input)

        cache_manager.vector_store.find_similar.assert_not_called()
        cache_manager.vector_store.store.assert_not_called()
        cache_manager.provider.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_get_response_cache_hit(cache_manager):
    """Test getting a response from cache."""
    # Mock cache hit
    cache_entry = CacheEntry(
        question="test question",
        answer="cached answer",
        provider="mock",
        timestamp=1234567890,
        embedding=[0.1] * 1536,
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
        prompt="test question", system_prompt=None, temperature=0.7, max_tokens=None
    )


@pytest.mark.asyncio
async def test_get_response_with_system_prompt(cache_manager):
    """Test getting a response with a custom system prompt."""
    # Mock cache miss
    cache_manager.vector_store.find_similar = AsyncMock(return_value=[])
    cache_manager.vector_store.store = AsyncMock()
    cache_manager.provider.generate_response.return_value = (
        "Generated response with system prompt"
    )

    # Get response with system prompt
    system_prompt = "Custom system prompt"
    response, from_cache, cache_info = await cache_manager.get_response(
        "test question", system_prompt=system_prompt
    )

    # Verify system prompt was passed to provider
    assert response == "Generated response with system prompt"
    assert from_cache is False
    assert cache_info is None
    cache_manager.provider.generate_response.assert_called_once_with(
        prompt="test question",
        system_prompt=system_prompt,
        temperature=0.7,
        max_tokens=None,
    )


@pytest.mark.asyncio
async def test_get_response_multiple_similar(cache_manager):
    """Test handling of multiple similar cached entries."""
    entries = [
        CacheEntry(
            question="first test question",
            answer="first answer",
            provider="mock",
            timestamp=1234567890,
            embedding=[0.1] * 1536,
        ),
        CacheEntry(
            question="second test question",
            answer="second answer",
            provider="mock",
            timestamp=1234567891,
            embedding=[0.2] * 1536,
        ),
    ]

    # Return multiple entries with different similarity scores
    cache_manager.vector_store.find_similar.return_value = [
        (entries[0], 0.95),
        (entries[1], 0.90),
    ]

    # Should return the most similar entry
    response, from_cache, cache_info = await cache_manager.get_response("test question")

    assert response == "first answer"
    assert from_cache is True
    assert cache_info == ("first test question", 0.95)
    cache_manager.vector_store.find_similar.assert_called_once()


@pytest.mark.asyncio
async def test_similarity_threshold_impact(cache_manager):
    """Test different similarity thresholds affect cache hits."""
    cache_entry = CacheEntry(
        question="test question",
        answer="cached answer",
        provider="mock",
        timestamp=1234567890,
        embedding=[0.1] * 1536,
    )

    # Test with similarity just below threshold
    cache_manager.config.cache.similarity_threshold = 0.9
    cache_manager.vector_store.find_similar.return_value = [(cache_entry, 0.85)]

    response, from_cache, cache_info = await cache_manager.get_response("test question")

    # Should miss cache due to low similarity
    assert from_cache is False
    assert cache_info is None

    # Test with similarity above threshold
    cache_manager.vector_store.find_similar.return_value = [(cache_entry, 0.95)]

    response, from_cache, cache_info = await cache_manager.get_response("test question")

    # Should hit cache
    assert from_cache is True
    assert cache_info == ("test question", 0.95)


@pytest.mark.asyncio
async def test_get_response_long_question(cache_manager):
    """Test handling of very long questions."""
    # Create a long question that would exceed typical token limits
    long_question = "test " * 1000  # 5000 characters

    # Mock the token count to simulate exceeding limits
    cache_manager.provider.get_token_count = MagicMock(return_value=2000)
    cache_manager.config.provider.max_tokens = 1000

    with pytest.raises(ValueError, match="Question exceeds maximum token limit"):
        await cache_manager.get_response(long_question)

    cache_manager.vector_store.find_similar.assert_not_called()
    cache_manager.vector_store.store.assert_not_called()
    cache_manager.provider.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup(cache_manager):
    """Test cache cleanup."""
    # Mock cleanup to return number of entries removed
    expected_removed = 5
    cache_manager.vector_store.cleanup_old_entries.return_value = expected_removed

    # Run cleanup
    removed = await cache_manager.cleanup()

    # Verify cleanup was called and returned correct number
    assert removed == expected_removed
    cache_manager.vector_store.cleanup_old_entries.assert_called_once_with()

    # Verify no other vector store methods were called
    cache_manager.vector_store.find_similar.assert_not_called()
    cache_manager.vector_store.store.assert_not_called()
