"""RAG (Retrieval-Augmented Generation) tools."""

from typing import List

from .schema import ToolParameter, ToolRegistry, ToolResult, RiskTier
from ..config.schema import RAGConfig
from ..connectors.qdrant import QdrantConnector
from ..services.embedding import EmbeddingService


class RAGTools:
    """RAG tool implementations."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_connector: QdrantConnector,
        rag_config: RAGConfig,
    ):
        """Initialize RAG tools.

        Args:
            embedding_service: Service for generating embeddings
            qdrant_connector: Connector for Qdrant vector store
            rag_config: RAG configuration
        """
        self.embedding_service = embedding_service
        self.qdrant_connector = qdrant_connector
        self.rag_config = rag_config

    async def rag_search(
        self,
        query: str,
        limit: int | None = None,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ) -> ToolResult:
        """Search for relevant document chunks.

        Returns formatted results with source citations.
        If no documents are ingested, returns empty results.

        Args:
            query: Natural language search query
            limit: Maximum number of results (default: from config)
            score_threshold: Minimum similarity score 0.0-1.0
            source_filter: Optional source path filter

        Returns:
            ToolResult with list of relevant chunks and metadata
        """
        try:
            # Use config defaults if not specified
            limit = limit or self.rag_config.top_k
            score_threshold = score_threshold or self.rag_config.score_threshold

            # Generate query embedding
            query_vector = await self.embedding_service.embed_text(query)

            # Search Qdrant
            results = self.qdrant_connector.search(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                source_filter=source_filter,
            )

            # Format results for LLM consumption
            formatted_results = []
            for r in results:
                payload = r["payload"]
                formatted_results.append(
                    {
                        "text": payload["text"],
                        "source": payload["source"],
                        "chunk_index": payload["chunk_index"],
                        "score": round(r["score"], 3),
                    }
                )

            # Format metadata for display
            metadata = {
                "query": query,
                "results_count": len(results),
                "limit": limit,
                "score_threshold": score_threshold,
            }

            # If no results, return empty (LLM will use base knowledge)
            if not results:
                return ToolResult(success=True, result=[], metadata=metadata)

            return ToolResult(
                success=True, result=formatted_results, metadata=metadata
            )

        except Exception as e:
            return ToolResult(
                success=False, error=f"RAG search failed: {str(e)}", metadata={"query": query}
            )


def register_rag_tools(
    registry: ToolRegistry,
    embedding_service: EmbeddingService,
    qdrant_connector: QdrantConnector,
    rag_config: RAGConfig,
) -> None:
    """Register RAG tools with the registry.

    Args:
        registry: Tool registry
        embedding_service: Embedding service instance
        qdrant_connector: Qdrant connector instance
        rag_config: RAG configuration
    """
    tools = RAGTools(embedding_service, qdrant_connector, rag_config)

    registry.register(
        name="rag_search",
        description=(
            "Search ingested documents for relevant information using semantic similarity. "
            "Returns chunks of text with source citations and similarity scores. "
            "Use this when you need to ground responses in local knowledge (code, docs, notes). "
            "If no results are found, use your base knowledge and inform the user."
        ),
        risk_tier=RiskTier.TIER_0,  # Read-only, no approval required
        handler=tools.rag_search,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Natural language search query",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of results (default: from config)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="score_threshold",
                type="number",
                description="Minimum similarity score 0.0-1.0",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="source_filter",
                type="string",
                description="Filter results by source path",
                required=False,
                default=None,
            ),
        ],
    )
