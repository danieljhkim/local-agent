"""Embedding service using Ollama."""

import asyncio
from typing import List
import httpx

from ..config.schema import EmbeddingConfig


class EmbeddingService:
    """Service for generating embeddings using Ollama."""

    def __init__(self, embedding_config: EmbeddingConfig):
        """Initialize embedding service.

        Args:
            embedding_config: Embedding configuration
        """
        self.config = embedding_config
        self.model = embedding_config.model
        self.base_url = embedding_config.base_url
        self.batch_size = embedding_config.batch_size
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ConnectionError: If unable to connect to Ollama
            ValueError: If model not found
            httpx.HTTPStatusError: If request fails
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)

        url = f"{self.base_url}/api/embeddings"

        try:
            response = await self._client.post(
                url, json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? (ollama serve)"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                ) from e
            raise

        data = response.json()
        return data["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ConnectionError: If unable to connect to Ollama
            ValueError: If model not found
        """
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Process batch in parallel
            tasks = [self.embed_text(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)

        return embeddings
