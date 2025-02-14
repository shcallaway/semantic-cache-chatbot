"""
Vector store implementation using Qdrant.
"""
import time
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models

from chatbot.cache.base import BaseVectorStore, CacheEntry
from chatbot.config import CacheConfig


class QdrantVectorStore(BaseVectorStore):
    """Vector store implementation using Qdrant."""

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
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(config.index_name)
        except Exception:
            self.client.create_collection(
                collection_name=config.index_name,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=models.Distance.COSINE,
                ),
            )

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
        timestamp = time.time()

        # Store in Qdrant
        self.client.upsert(
            collection_name=self.config.index_name,
            points=[
                models.PointStruct(
                    id=hash(question),
                    vector=embedding,
                    payload={
                        "question": question,
                        "answer": answer,
                        "provider": provider,
                        "timestamp": timestamp,
                        "namespace": self.config.namespace,
                    },
                )
            ],
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

        # Query Qdrant
        results = self.client.search(
            collection_name=self.config.index_name,
            query_vector=embedding,
            limit=limit,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(value=self.config.namespace),
                    )
                ]
            ),
        )

        # Convert results to cache entries
        entries = []
        for match in results:
            if match.score >= self.config.similarity_threshold:
                entry = CacheEntry(
                    question=match.payload["question"],
                    answer=match.payload["answer"],
                    provider=match.payload["provider"],
                    timestamp=match.payload["timestamp"],
                    embedding=match.vector,
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
        result = self.client.delete(
            collection_name=self.config.index_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(value=self.config.namespace),
                    ),
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(lt=cutoff_time),
                    ),
                ]
            ),
        )
        
        return result.deleted_count
