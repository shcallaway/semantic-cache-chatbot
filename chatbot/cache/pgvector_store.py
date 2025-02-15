"""
Vector store implementation using PostgreSQL with pgvector extension.
"""

import time
from typing import List, Optional, Tuple

from openai import AsyncOpenAI
import psycopg2
from psycopg2.extras import DictCursor
from pgvector.psycopg2 import register_vector

from chatbot.cache.base import BaseVectorStore, CacheEntry
from chatbot.config import CacheConfig


class PgVectorStore(BaseVectorStore):
    """Vector store implementation using PostgreSQL with pgvector extension."""

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

        # Initialize PostgreSQL connection parameters
        self.conn_params = {
            "host": config.postgres_host,
            "port": config.postgres_port or 5432,
            "user": config.postgres_user,
            "password": config.postgres_password,
            "database": config.postgres_db,
            "cursor_factory": DictCursor,
        }

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize the database schema and extensions."""
        conn = psycopg2.connect(**self.conn_params)
        try:
            cur = conn.cursor()
            # Create vector extension if it doesn't exist
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create cache table if it doesn't exist
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chatbot (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL,
                    embedding vector(1536) NOT NULL
                )
            """
            )

            # Create index for vector similarity search
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chatbot_embedding_idx 
                ON chatbot 
                USING ivfflat (embedding vector_cosine_ops)
            """
            )

            conn.commit()
            cur.close()
        finally:
            conn.close()

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

        timestamp = time.time()

        conn = psycopg2.connect(**self.conn_params)
        try:
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chatbot
                (question, answer, provider, namespace, timestamp, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    question,
                    answer,
                    provider,
                    self.config.namespace,
                    timestamp,
                    embedding,
                ),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

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

        conn = psycopg2.connect(**self.conn_params)
        try:
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 
                    question,
                    answer,
                    provider,
                    timestamp,
                    embedding,
                    1 - (embedding <=> %s::vector) as similarity
                FROM chatbot
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, self.config.namespace, embedding, limit),
            )
            results = cur.fetchall()
            cur.close()

            entries = []
            for row in results:
                if (
                    row[5] >= self.config.similarity_threshold
                ):  # similarity is at index 5
                    entry = CacheEntry(
                        question=row[0],  # question
                        answer=row[1],  # answer
                        provider=row[2],  # provider
                        timestamp=row[3],  # timestamp
                        embedding=row[4],  # embedding
                    )
                    entries.append((entry, row[5]))  # row[5] is similarity

            return entries
        finally:
            conn.close()

    async def cleanup_old_entries(self) -> int:
        """Remove entries older than the TTL.

        Returns:
            Number of entries removed
        """
        try:
            print(f"Starting cleanup for pgvector collection: {self.config.index_name}")
            print(f"Namespace: {self.config.namespace}")
            print(
                f"TTL days: {self.config.ttl_days if self.config.ttl_days is not None else 'None (deleting all)'}"
            )

            conn = psycopg2.connect(**self.conn_params)
            try:
                cur = conn.cursor()
                if self.config.ttl_days is None or self.config.ttl_days <= 0:
                    print("No TTL specified - will remove all entries in namespace")
                    cur.execute(
                        "DELETE FROM chatbot WHERE namespace = %s",
                        (self.config.namespace,),
                    )
                else:
                    cutoff_time = time.time() - (self.config.ttl_days * 24 * 60 * 60)
                    print(f"Will remove entries older than: {time.ctime(cutoff_time)}")
                    cur.execute(
                        """
                        DELETE FROM chatbot 
                        WHERE namespace = %s AND timestamp < %s
                        """,
                        (self.config.namespace, cutoff_time),
                    )

                deleted_count = cur.rowcount
                conn.commit()
                cur.close()
                print(f"Deleted {deleted_count} entries")
                return deleted_count
            finally:
                conn.close()

        except Exception as e:
            print(f"Warning: Cache cleanup failed - {str(e)}")
            return 0
