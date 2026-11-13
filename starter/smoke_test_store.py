
import os
from pathlib import Path

from dotenv import load_dotenv

from starter.src.models import Chunk
from starter.src.store import PgVectorStore


load_dotenv(Path("starter/.env"))

database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is missing.")

store = PgVectorStore(database_url, dimensions=768)
store.create_schema()

# Synthetic vector for testing database operations only.
vector = [0.0] * 768
vector[0] = 1.0

chunk = Chunk(
    chunk_id="smoke-test-p1-fixed-0000",
    document_id="smoke-test-doc",
    source="Synthetic test data",
    page=1,
    text="Synthetic content for pgvector smoke testing.",
    metadata={
        "chapter": "1",
        "section": "1.1",
        "category": "chapter",
        "language": "en",
    },
)

try:
    store.upsert("A", [chunk], [vector])

    hits = store.search(
        strategy="A",
        query_vector=vector,
        top_k=1,
        filters={"chapter": "1"},
    )

    assert len(hits) == 1, "Expected one search result."
    assert hits[0]["chunk_id"] == chunk.chunk_id
    assert hits[0]["metadata"]["chapter"] == "1"

    print("Database connection: PASS")
    print("Vector insertion: PASS")
    print("Top-1 vector search: PASS")
    print("Metadata filter: PASS")
    print("Similarity score:", round(hits[0]["score"], 4))

finally:
    # Remove only the synthetic test record.
    with store._connect() as conn:
        conn.execute(
            """
            DELETE FROM document_chunks
            WHERE strategy = %s AND document_id = %s
            """,
            ("A", "smoke-test-doc"),
        )

    print("Synthetic test record cleanup: PASS")
