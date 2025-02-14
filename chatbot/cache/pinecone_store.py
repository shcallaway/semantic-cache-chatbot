"""
Vector store implementation using Pinecone.
"""

import time
from time import sleep
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
from pinecone import Pinecone

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

        # Initialize Pinecone with API key only
        self.pc = Pinecone(api_key=config.pinecone_api_key)

        # List available indexes and create if needed
        indexes = self.pc.list_indexes()
        index_exists = any(idx.name == config.index_name for idx in indexes)

        if not index_exists:
            self.pc.create_index(
                name=config.index_name,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )
            # Wait for index to be ready
            sleep(10)

        # Get index instance
        self.index = self.pc.Index(config.index_name)

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
                        "namespace": self.config.namespace,
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
        for match in results["matches"]:
            if match["score"] >= self.config.similarity_threshold:
                entry = CacheEntry(
                    question=match["metadata"]["question"],
                    answer=match["metadata"]["answer"],
                    provider=match["metadata"]["provider"],
                    timestamp=match["metadata"]["timestamp"],
                    embedding=match["values"],
                )
                entries.append((entry, match["score"]))

        return entries

    def get_total_entries(self) -> int:
        """Get total number of entries in the vector store.

        Returns:
            Total number of entries
        """
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"Raw index stats: {stats}")

            # Check if stats has the expected structure
            if hasattr(stats, "total_vector_count"):
                return stats.total_vector_count
            elif hasattr(stats, "namespaces"):
                # Sum up vectors across all namespaces
                total = sum(ns.vector_count for ns in stats.namespaces.values())
                print(f"Calculated total from namespaces: {total}")
                return total
            else:
                # Try to access as dictionary
                if isinstance(stats, dict):
                    if "total_vector_count" in stats:
                        return stats["total_vector_count"]
                    elif "namespaces" in stats:
                        total = sum(
                            ns.get("vector_count", 0)
                            for ns in stats["namespaces"].values()
                        )
                        print(f"Calculated total from dict namespaces: {total}")
                        return total

                print(f"Unknown stats structure: {type(stats)}")
                return 0

        except Exception as e:
            print(f"Warning: Failed to get entry count - {str(e)}")
            print(f"Error type: {type(e)}")
            if hasattr(e, "response"):
                print(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
                print(f"Response body: {getattr(e.response, 'text', 'N/A')}")
            return 0

    async def cleanup_old_entries(self) -> int:
        """Remove entries older than the TTL.

        Returns:
            Number of entries removed
        """
        try:
            print(f"Starting cleanup for Pinecone index: {self.config.index_name}")
            print(f"Namespace: {self.config.namespace}")
            print(f"TTL days: {self.config.ttl_days if self.config.ttl_days is not None else 'None (deleting all)'}")

            # Get initial count
            stats = self.index.describe_index_stats()
            if hasattr(stats, "namespaces"):
                namespaces = stats.namespaces
            elif isinstance(stats, dict) and "namespaces" in stats:
                namespaces = stats["namespaces"]
            else:
                print("Warning: Unable to get namespace statistics")
                return 0

            namespace_stats = namespaces.get(self.config.namespace, {})
            initial_count = namespace_stats.get("vector_count", 0)
            print(f"Found {initial_count} entries in namespace")

            if initial_count == 0:
                return 0

            if self.config.ttl_days is None or self.config.ttl_days <= 0:
                print("No TTL specified - will remove all entries in namespace")
                try:
                    # Try direct namespace deletion
                    self.index.delete(namespace=self.config.namespace, delete_all=True)
                    print("Deletion complete")
                    return initial_count
                except Exception:
                    # Fall back to batch deletion
                    print("Falling back to batch deletion...")
                    query_response = self.index.query(
                        vector=[1.0 / 1536] * 1536,
                        top_k=initial_count,
                        namespace=self.config.namespace,
                        include_metadata=False,
                    )

                    if not hasattr(query_response, "matches"):
                        return 0

                    vector_ids = [match.id for match in query_response.matches]
                    if not vector_ids:
                        return 0

                    # Delete in batches
                    batch_size = 100
                    for i in range(0, len(vector_ids), batch_size):
                        batch = vector_ids[i : i + batch_size]
                        self.index.delete(ids=batch, namespace=self.config.namespace)
                    print("Deletion complete")
                    return len(vector_ids)
            else:
                # Delete old entries
                cutoff_time = time.time() - (self.config.ttl_days * 24 * 60 * 60)
                print(f"Will remove entries older than: {time.ctime(cutoff_time)}")

                query_response = self.index.query(
                    vector=[1.0 / 1536] * 1536,
                    top_k=initial_count,
                    namespace=self.config.namespace,
                    include_metadata=True,
                )

                if not hasattr(query_response, "matches"):
                    return 0

                old_vector_ids = [
                    match.id
                    for match in query_response.matches
                    if match.metadata
                    and "timestamp" in match.metadata
                    and float(match.metadata["timestamp"]) < cutoff_time
                ]

                if not old_vector_ids:
                    return 0

                print(f"Found {len(old_vector_ids)} entries to remove")
                
                # Delete in batches
                batch_size = 100
                for i in range(0, len(old_vector_ids), batch_size):
                    batch = old_vector_ids[i : i + batch_size]
                    self.index.delete(ids=batch, namespace=self.config.namespace)
                print("Deletion complete")
                return len(old_vector_ids)

        except Exception as e:
            print(f"Warning: Cache cleanup failed - {str(e)}")
            return 0
