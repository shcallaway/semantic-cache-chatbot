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

        # List available indexes first
        indexes = self.pc.list_indexes()
        print(
            f"Available indexes before initialization: {[idx.name for idx in indexes]}"
        )

        # Check if index exists
        index_exists = any(idx.name == config.index_name for idx in indexes)

        if not index_exists:
            print(f"Creating index: {config.index_name}")
            self.pc.create_index(
                name=config.index_name,
                dimension=1536,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            )
            # Wait for index to be ready
            sleep(10)
            print("Index created successfully")

        # Get index instance
        self.index = self.pc.Index(config.index_name)
        print("Index connection established")

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
            # Get initial stats
            stats = self.index.describe_index_stats()
            print(f"Initial stats: {stats}")

            # Handle both object and dictionary formats
            if hasattr(stats, "namespaces"):
                namespaces = stats.namespaces
            elif isinstance(stats, dict) and "namespaces" in stats:
                namespaces = stats["namespaces"]
            else:
                print(f"Unable to get namespaces from stats: {stats}")
                return 0

            print(f"Namespaces: {namespaces}")

            # Get namespace stats
            if hasattr(namespaces, "get"):
                namespace_stats = namespaces.get(self.config.namespace, {})
            elif isinstance(namespaces, dict):
                namespace_stats = namespaces.get(self.config.namespace, {})
            else:
                print(f"Invalid namespaces type: {type(namespaces)}")
                return 0

            print(f"Namespace stats: {namespace_stats}")
            initial_count = namespace_stats.get("vector_count", 0)
            print(f"Initial vector count: {initial_count}")

            if initial_count == 0:
                print("No vectors to clean up")
                return 0

            try:
                print(f"TTL days: {self.config.ttl_days}")
                if self.config.ttl_days is None or self.config.ttl_days <= 0:
                    print("Attempting full deletion...")
                    try:
                        # Try direct namespace deletion first
                        print(
                            f"Attempting to delete namespace: {self.config.namespace}"
                        )
                        self.index.delete(
                            namespace=self.config.namespace, delete_all=True
                        )
                        print("Namespace deletion successful")
                        return initial_count
                    except Exception as e:
                        print(f"Namespace deletion failed: {e}")
                        print("Falling back to vector-by-vector deletion...")

                        # Fall back to vector-by-vector deletion
                        print("Fetching all vectors...")
                        query_response = self.index.query(
                            vector=[1.0 / 1536] * 1536,  # Normalized vector
                            top_k=initial_count,  # Get all vectors
                            namespace=self.config.namespace,
                            include_metadata=False,  # Don't need metadata for full deletion
                        )
                        print(f"Query response: {query_response}")

                        if not hasattr(query_response, "matches"):
                            print("No vectors found in query response")
                            return 0

                        vector_ids = [match.id for match in query_response.matches]
                        print(f"Found {len(vector_ids)} vectors")

                        if not vector_ids:
                            print("No vectors to delete")
                            return 0

                        print(f"Deleting all {len(vector_ids)} vectors...")
                        try:
                            # Delete in smaller batches
                            batch_size = 100
                            for i in range(0, len(vector_ids), batch_size):
                                batch = vector_ids[i : i + batch_size]
                                print(
                                    f"Deleting batch {i//batch_size + 1} of {(len(vector_ids)-1)//batch_size + 1}..."
                                )
                                self.index.delete(
                                    ids=batch, namespace=self.config.namespace
                                )
                            print("All vectors deleted successfully")
                            return len(vector_ids)
                        except Exception as e:
                            print(f"Error during batch deletion: {e}")
                            raise
                else:
                    # Get vectors with timestamp filter
                    cutoff_time = time.time() - (self.config.ttl_days * 24 * 60 * 60)
                    print(f"Fetching vectors older than: {cutoff_time}")

                    query_response = self.index.query(
                        vector=[1.0 / 1536] * 1536,  # Normalized vector
                        top_k=initial_count,  # Get all vectors
                        namespace=self.config.namespace,
                        include_metadata=True,
                    )
                    print(f"Query response: {query_response}")

                    if not hasattr(query_response, "matches"):
                        print("No vectors found in query response")
                        return 0

                    old_vector_ids = []
                    for match in query_response.matches:
                        if match.metadata and "timestamp" in match.metadata:
                            if float(match.metadata["timestamp"]) < cutoff_time:
                                old_vector_ids.append(match.id)

                    if old_vector_ids:
                        print(f"Deleting {len(old_vector_ids)} old vectors...")
                        try:
                            # Delete in smaller batches
                            batch_size = 100
                            for i in range(0, len(old_vector_ids), batch_size):
                                batch = old_vector_ids[i : i + batch_size]
                                print(
                                    f"Deleting batch {i//batch_size + 1} of {(len(old_vector_ids)-1)//batch_size + 1}..."
                                )
                                self.index.delete(
                                    ids=batch, namespace=self.config.namespace
                                )
                            print("Old vectors deleted successfully")
                            return len(old_vector_ids)
                        except Exception as e:
                            print(f"Error during batch deletion: {e}")
                            raise
                    else:
                        print("No vectors old enough to delete")
                        return 0

            except Exception as e:
                print(f"Delete operation failed: {str(e)}")
                print(f"Error type: {type(e)}")
                if hasattr(e, "response"):
                    print(
                        f"Response status: {getattr(e.response, 'status_code', 'N/A')}"
                    )
                    print(f"Response body: {getattr(e.response, 'text', 'N/A')}")
                raise

        except Exception as e:
            print(f"Warning: Cache cleanup failed - {str(e)}")
            return 0
