
"""PostgreSQL + pgvector storage and retrieval."""
import math
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import Chunk


# Only these metadata fields can be used as retrieval filters.
ALLOWED_FILTERS = {
    "chapter",
    "section",
    "year",
    "category",
    "audience",
    "language",
    "version",
}


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required.")

        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer.")

        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        return psycopg.connect(
            self.database_url,
            row_factory=dict_row,
        )

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        """Convert a numeric vector to pgvector's textual representation."""
        values = []

        for value in vector:
            number = float(value)

            if not math.isfinite(number):
                raise ValueError("Vector contains a non-finite value.")

            values.append(str(number))

        return "[" + ",".join(values) + "]"

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} dimensions, "
                f"received {len(vector)}."
            )

        self._vector_literal(vector)

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                document_chunks_strategy_idx
                ON document_chunks (strategy)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                document_chunks_metadata_idx
                ON document_chunks USING GIN (metadata)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                document_chunks_embedding_idx
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if not strategy:
            raise ValueError("strategy cannot be empty.")

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks and vectors must match."
            )

        if not chunks:
            return

        document_ids = set()

        for chunk, vector in zip(chunks, vectors):
            if not chunk.chunk_id:
                raise ValueError("chunk_id cannot be empty.")

            if not chunk.document_id:
                raise ValueError("document_id cannot be empty.")

            if not chunk.source:
                raise ValueError("source cannot be empty.")

            if not isinstance(chunk.page, int) or chunk.page < 1:
                raise ValueError("page must be a positive integer.")

            if not chunk.text.strip():
                raise ValueError("Chunk content cannot be empty.")

            if not isinstance(chunk.metadata, dict):
                raise ValueError("Chunk metadata must be a dictionary.")

            self._validate_vector(vector)
            document_ids.add(chunk.document_id)

        rows = [
            (
                strategy,
                chunk.chunk_id,
                chunk.document_id,
                chunk.source,
                chunk.page,
                chunk.text,
                Jsonb(chunk.metadata),
                self._vector_literal(vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        with self._connect() as conn:
            # Replace previous chunks for the affected documents.
            # This also removes obsolete chunks after re-ingestion.
            conn.execute(
                """
                DELETE FROM document_chunks
                WHERE strategy = %s
                  AND document_id = ANY(%s)
                """,
                (strategy, list(document_ids)),
            )

            conn.cursor().executemany(
                """
                INSERT INTO document_chunks (
                    strategy,
                    chunk_id,
                    document_id,
                    source,
                    page,
                    content,
                    metadata,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s::vector
                )
                """,
                rows,
            )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not strategy:
            raise ValueError("strategy cannot be empty.")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        self._validate_vector(query_vector)

        filters = filters or {}

        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary.")

        invalid_keys = set(filters) - ALLOWED_FILTERS

        if invalid_keys:
            raise ValueError(
                f"Unsupported filter fields: {sorted(invalid_keys)}"
            )

        sql = """
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                content AS text,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE strategy = %s
        """

        params: list[Any] = [
            self._vector_literal(query_vector),
            strategy,
        ]

        if filters:
            sql += " AND metadata @> %s::jsonb"
            params.append(Jsonb(filters))

        sql += """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params.extend([
            self._vector_literal(query_vector),
            top_k,
        ])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]
