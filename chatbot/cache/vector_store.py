"""
Vector store implementation using Pinecone.
"""
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pinecone
from openai import AsyncOpenAI

from chatbot.config import CacheConfig


@dataclass
class CacheEntry:
    """Represents a cached question-answer pair."""
    question: str
    answer: str
    provider: str
    timestamp: float
    embedding: List[float]


class VectorStore:
    """Vector store implementation using Pinecone."""

    def __init__(
        self,
        config: CacheConfig,
        openai_client: AsyncOpenAI,
    ):
        """Initialize the vector store.

        Args:
            config: Cache configuration
            openai_client: OpenAI client for generating embeddings
        """
        self.config = config
        self.openai_client = openai_client
        
        # Initialize Pinecone
        pinecone.init(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
        )
        
        # Create index if it doesn't exist
        if config.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=config.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
            )
        
        self.index = pinecone.Index(config.index_name)

    async def store(
        self,
        question: str,
        answer: str,
        provider: str,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Store a question-answer pair in the cache.

        Args:
            question: The user's question
            answer: The model's answer
            provider: The LLM provider used
            embedding: Optional pre-computed embedding for the question
        """
        if embedding is None:
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=question,
            )
            embedding = embedding_response.data[0].embedding

        # Create cache entry
        entry = CacheEntry(
            question=question,
            answer=answer,
            provider=provider,
            timestamp=time.time(),
            embedding=embedding,
        )

        # Store in Pinecone
        self.index.upsert(
            vectors=[
                {
                    "id": str(hash(question)),
                    "values": embedding,
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "provider": provider,
                        "timestamp": entry.timestamp,
                    },
                }
            ],
            namespace=self.config.namespace,
        )

    async def find_similar(
        self,
        question: str,
        limit: int = 1,
    ) -> List[Tuple[CacheEntry, float]]:
        """Find similar questions in the cache.

        Args:
            question: The question to find similar entries for
            limit: Maximum number of results to return

        Returns:
            List of tuples containing cache entries and their similarity scores
        """
        # Generate embedding for the question
        embedding_response = await self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=question,
        )
        embedding = embedding_response.data[0].embedding

        # Query Pinecone
        results = self.index.query(
            vector=embedding,
            top_k=limit,
            namespace=self.config.namespace,
            include_metadata=True,
        )

        # Convert results to cache entries
        entries = []
        for match in results.matches:
            if match.score >= self.config.similarity_threshold:
                entry = CacheEntry(
                    question=match.metadata["question"],
                    answer=match.metadata["answer"],
                    provider=match.metadata["provider"],
                    timestamp=match.metadata["timestamp"],
                    embedding=match.values,
                )
                entries.append((entry, match.score))

        return entries

    def cleanup_old_entries(self) -> int:
        """Remove entries older than the TTL.

        Returns:
            Number of entries removed
        """
        cutoff_time = time.time() - (self.config.ttl_days * 24 * 60 * 60)
        
        # Delete old vectors
        delete_response = self.index.delete(
            filter={
                "timestamp": {"$lt": cutoff_time}
            },
            namespace=self.config.namespace,
        )
        
        return delete_response.deleted_count if hasattr(delete_response, 'deleted_count') else 0
