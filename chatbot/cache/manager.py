"""
Cache manager for coordinating between vector store and LLM providers.
"""
from typing import Optional, Tuple

from openai import AsyncOpenAI

from chatbot.cache.vector_store import CacheEntry, VectorStore
from chatbot.config import Config
from chatbot.providers.base import LLMProvider


class CacheManager:
    """Manages semantic caching for LLM responses."""

    def __init__(
        self,
        config: Config,
        openai_client: AsyncOpenAI,
        provider: LLMProvider,
    ):
        """Initialize the cache manager.

        Args:
            config: Application configuration
            openai_client: OpenAI client for embeddings
            provider: LLM provider instance
        """
        self.config = config
        self.vector_store = VectorStore(config.cache, openai_client)
        self.provider = provider

    async def get_response(
        self,
        question: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, bool, Optional[Tuple[str, float]]]:
        """Get a response for the question, using cache if available.

        Args:
            question: The user's question
            system_prompt: Optional system prompt for the LLM

        Returns:
            Tuple of (response text, whether it was from cache, optional tuple of (matched question, similarity score))
        """
        # Try to find similar questions in cache
        similar_entries = await self.vector_store.find_similar(question)

        # If we found a similar enough question, use its cached answer
        if similar_entries:
            entry, similarity = similar_entries[0]
            return entry.answer, True, (entry.question, similarity)

        # Otherwise, generate a new response
        response = await self.provider.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            temperature=self.config.provider.temperature,
            max_tokens=self.config.provider.max_tokens,
        )

        # Cache the new question-answer pair
        # We can pass the embedding from the similarity search to avoid recomputing
        await self.vector_store.store(
            question=question,
            answer=response,
            provider=self.provider.provider_name,
        )

        return response, False, None

    async def cleanup(self) -> int:
        """Clean up old cache entries.

        Returns:
            Number of entries removed
        """
        return self.vector_store.cleanup_old_entries()
