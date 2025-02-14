"""
Vector store implementation using Pinecone.
"""
import time
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
from pinecone import Pinecone
from pinecone import PineconeException

from chatbot.cache.base import BaseVectorStore, CacheEntry
from chatbot.config import CacheConfig


class PineconeVectorStore(BaseVectorStore):
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
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        try:
            # Try to get the index if it exists
            self.index = self.pc.Index(config.index_name)
        except PineconeException as e:
            if "index not found" in str(e).lower():
                # Only try to create if index doesn't exist
                try:
                    self.pc.create_index(
                        name=config.index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric="cosine",
                        spec={
                            "pod": {
                                "environment": "us-east-1-aws",
                                "pod_type": "starter"
                            }
                        }
                    )
                    self.index = self.pc.Index(config.index_name)
                except PineconeException as create_error:
                    if "max pods allowed" in str(create_error).lower():
                        raise RuntimeError(
                            "Unable to create new Pinecone index - maximum pods reached. "
                            "Please delete unused indexes or upgrade your plan."
                        ) from create_error
                    raise
            else:
                raise

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
            include_values=True,
        )

        # Convert results to cache entries
        entries = []
        for match in results['matches']:
            if match['score'] >= self.config.similarity_threshold:
                entry = CacheEntry(
                    question=match['metadata']["question"],
                    answer=match['metadata']["answer"],
                    provider=match['metadata']["provider"],
                    timestamp=match['metadata']["timestamp"],
                    embedding=match['values'],
                )
                entries.append((entry, match['score']))

        return entries

    def get_total_entries(self) -> int:
        """Get total number of entries in the vector store.

        Returns:
            Total number of entries
        """
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            print(f"Warning: Failed to get entry count - {str(e)}")
            return 0

    def cleanup_old_entries(self) -> int:
        """Remove entries older than the TTL.

        Returns:
            Number of entries removed
        """
        # Get initial count
        initial_count = self.get_total_entries()
        print(f"Current number of entries: {initial_count}")
        
        try:
            if self.config.ttl_days == 0:
                # If TTL is 0, delete all entries
                self.index.delete(
                    delete_all=True,
                    namespace=self.config.namespace
                )
                final_count = self.get_total_entries()
                deleted = initial_count - final_count
                print(f"Successfully cleaned up all {deleted} entries")
                print(f"Total entries: {initial_count} -> {final_count}")
                return deleted
                
            # Otherwise, only clean up old entries
            cutoff_time = time.time() - (self.config.ttl_days * 24 * 60 * 60)
            query_response = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                top_k=10000,
                namespace=self.config.namespace,
                filter={"timestamp": {"$lt": cutoff_time}},
                include_metadata=True
            )
            
            # Get IDs of old vectors
            old_ids = [match['id'] for match in query_response['matches']]
            
            if not old_ids:
                print("No old entries found to clean up")
                return 0
                
            # Delete the old vectors by ID
            delete_response = self.index.delete(
                ids=old_ids,
                namespace=self.config.namespace
            )
            
            deleted = len(old_ids)
            
            # Get final count after cleanup
            final_count = self.get_total_entries()
            
            print(f"Successfully cleaned up {deleted} old cache entries")
            print(f"Total entries: {initial_count} -> {final_count}")
            
            return deleted
            
        except Exception as e:
            print(f"Warning: Cache cleanup failed - {str(e)}")
            return 0
