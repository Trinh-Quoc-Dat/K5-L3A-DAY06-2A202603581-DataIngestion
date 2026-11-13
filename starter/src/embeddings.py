
"""Gemini embedding adapter with batching and usage statistics."""

import os
import time
from google import genai
from google.genai import types


class MyEmbedder:
    def __init__(
        self,
        model_name: str,
        dimensions: int = 768,
        batch_size: int = 32,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")

        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")

        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100.")

        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.client = genai.Client(api_key=api_key)

        self.stats = {
            "document_requests": 0,
            "query_requests": 0,
            "document_texts": 0,
            "query_texts": 0,
            "document_seconds": 0.0,
            "query_seconds": 0.0,
            "total_tokens": 0,
            "token_usage_available": False,
        }

    def _embed_batch(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Embedding input contains empty text.")

        started = time.perf_counter()

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions,
            ),
        )

        elapsed = time.perf_counter() - started

        if task_type == "RETRIEVAL_DOCUMENT":
            self.stats["document_requests"] += 1
            self.stats["document_texts"] += len(texts)
            self.stats["document_seconds"] += elapsed
        elif task_type == "RETRIEVAL_QUERY":
            self.stats["query_requests"] += 1
            self.stats["query_texts"] += len(texts)
            self.stats["query_seconds"] += elapsed

        # Some embedding responses may expose usage metadata.
        usage = getattr(response, "usage_metadata", None)

        if usage is not None:
            token_count = getattr(
                usage,
                "total_token_count",
                None,
            )

            if token_count is not None:
                self.stats["total_tokens"] += int(token_count)
                self.stats["token_usage_available"] = True

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")

        vectors = [
            embedding.values
            for embedding in response.embeddings
        ]

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} vectors, got {len(vectors)}."
            )

        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Vector {index} has {len(vector)} dimensions; "
                    f"expected {self.dimensions}."
                )

        return vectors

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        vectors = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            batch_vectors = self._embed_batch(
                batch,
                task_type="RETRIEVAL_DOCUMENT",
            )

            vectors.extend(batch_vectors)

        return vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = self._embed_batch(
            [question],
            task_type="RETRIEVAL_QUERY",
        )

        return vectors[0]

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "document_seconds": round(
                self.stats["document_seconds"], 3
            ),
            "query_seconds": round(
                self.stats["query_seconds"], 3
            ),
            "total_seconds": round(
                self.stats["document_seconds"]
                + self.stats["query_seconds"],
                3,
            ),
        }
