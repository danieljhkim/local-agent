"""Qdrant vector database connector."""

from typing import Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from ..config.schema import QdrantConfig


class QdrantConnector:
    """Sandboxed Qdrant vector database connector."""

    def __init__(self, qdrant_config: QdrantConfig):
        """Initialize Qdrant connector.

        Args:
            qdrant_config: Qdrant configuration

        Raises:
            ConnectionError: If unable to connect to Qdrant
        """
        self.config = qdrant_config
        self.client = QdrantClient(url=qdrant_config.url, timeout=qdrant_config.timeout)
        self.collection_name = qdrant_config.collection_name
        self.vector_size = qdrant_config.vector_size

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist.

        Raises:
            Exception: If unable to create collection
        """
        try:
            collections = self.client.get_collections()
            exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self.config.url}. "
                "Is the Docker container running? (docker-compose up -d)"
            ) from e

    def upsert_points(
        self, points: List[PointStruct], batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vectors with batching.

        Args:
            points: List of points to upsert
            batch_size: Batch size for upsert operations

        Returns:
            Dict with upsert statistics

        Raises:
            Exception: If upsert fails
        """
        # Batch upserts to avoid memory issues
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        return {"upserted": len(points)}

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        source_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            source_filter: Optional source path filter

        Returns:
            List of search results with payload and score

        Raises:
            Exception: If search fails
        """
        filter_conditions = None
        if source_filter:
            filter_conditions = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source_filter))
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_conditions,
        )

        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]

    def delete_by_source(self, source_path: str) -> int:
        """Delete all points with matching source path.

        Args:
            source_path: Source file path to delete

        Returns:
            Number of points deleted (estimated)

        Raises:
            Exception: If deletion fails
        """
        # Qdrant delete by filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_path))]
            ),
        )
        # Note: Qdrant doesn't return count, will be tracked in SQLite
        return 0

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict with points_count and vectors_count

        Raises:
            Exception: If unable to get collection info
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count or info.points_count,
        }
