"""Document ingestion pipeline for RAG."""

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import chardet
import tiktoken
from qdrant_client.models import PointStruct
from sqlalchemy.orm import Session

from ..config.schema import RAGConfig
from ..connectors.qdrant import QdrantConnector
from ..persistence.db_models import Document
from .embedding import EmbeddingService


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    """Document ingestion pipeline for RAG."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_connector: QdrantConnector,
        rag_config: RAGConfig,
        db_session: Session | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            embedding_service: Service for generating embeddings
            qdrant_connector: Connector for Qdrant vector store
            rag_config: RAG configuration
            db_session: Optional database session for metadata tracking
        """
        self.embedding_service = embedding_service
        self.qdrant_connector = qdrant_connector
        self.rag_config = rag_config
        self.db_session = db_session
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def ingest_file(
        self, file_path: str, metadata: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Ingest a single file.

        Args:
            file_path: Path to file to ingest
            metadata: Optional additional metadata

        Returns:
            Ingestion statistics

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If ingestion fails
        """
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Load text
        text = self._load_text(path)

        # Compute hash for deduplication
        file_hash = self._compute_file_hash(path)

        # Check if already ingested
        if self.db_session:
            existing = (
                self.db_session.query(Document)
                .filter_by(source_path=str(path))
                .first()
            )

            if existing and existing.file_hash == file_hash:
                return {
                    "status": "skipped",
                    "reason": "unchanged",
                    "message": f"File unchanged since {existing.ingested_at}",
                }

            # File changed - delete old version
            if existing:
                self.qdrant_connector.delete_by_source(str(path))
                self.db_session.delete(existing)
                self.db_session.commit()

        # Chunk text
        chunks = self._chunk_text(text, str(path))

        if not chunks:
            return {
                "status": "skipped",
                "reason": "empty",
                "message": "File is empty or contains no text",
            }

        # Generate embeddings
        points = await self._embed_chunks(chunks)

        # Upsert to Qdrant
        self.qdrant_connector.upsert_points(points)

        # Record in database
        if self.db_session:
            doc = Document(
                id=str(uuid.uuid4()),
                source_path=str(path),
                file_hash=file_hash,
                file_size_bytes=path.stat().st_size,
                chunk_count=len(chunks),
                token_count=sum(c.token_count for c in chunks),
                file_type=path.suffix,
                collection_name=self.qdrant_connector.collection_name,
            )
            self.db_session.add(doc)
            self.db_session.commit()

        return {
            "status": "success",
            "chunks_created": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
        }

    async def ingest_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*",
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """Ingest all matching files in a directory.

        Args:
            directory_path: Root directory path
            glob_pattern: Glob pattern for files
            recursive: Whether to search recursively

        Returns:
            Aggregated ingestion statistics

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        path = Path(directory_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # Find matching files
        if recursive:
            files = list(path.glob(glob_pattern))
        else:
            files = [f for f in path.iterdir() if f.is_file()]

        # Filter by supported extensions
        supported = set(self.rag_config.supported_extensions)
        files = [f for f in files if f.suffix in supported and f.is_file()]

        # Ingest each file
        results = []
        for file in files:
            try:
                result = await self.ingest_file(str(file))
                results.append(result)
            except Exception as e:
                # Log error but continue with other files
                import traceback
                error_details = f"{type(e).__name__}: {str(e)}"
                traceback_str = traceback.format_exc()
                results.append(
                    {
                        "status": "error",
                        "file": str(file),
                        "error": error_details,
                        "traceback": traceback_str,
                    }
                )

        # Aggregate statistics
        successful = [r for r in results if r.get("status") == "success"]
        skipped = [r for r in results if r.get("status") == "skipped"]
        errors = [r for r in results if r.get("status") == "error"]

        result_dict = {
            "files_processed": len(files),
            "files_ingested": len(successful),
            "files_skipped": len(skipped),
            "files_errored": len(errors),
            "chunks_created": sum(r.get("chunks_created", 0) for r in successful),
            "total_tokens": sum(r.get("total_tokens", 0) for r in successful),
        }

        # Include first error for debugging
        if errors:
            result_dict["error_sample"] = {
                "file": errors[0].get("file"),
                "error": errors[0].get("error"),
                "traceback": errors[0].get("traceback"),
            }

        return result_dict

    def _load_text(self, file_path: Path) -> str:
        """Load text from file with encoding detection.

        Args:
            file_path: Path to file

        Returns:
            File contents as text

        Raises:
            Exception: If unable to read file
        """
        # Detect encoding
        with open(file_path, "rb") as f:
            raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected["encoding"] or "utf-8"

        # Load with detected encoding
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            return f.read()

    def _chunk_text(self, text: str, source: str) -> List[TextChunk]:
        """Chunk text into overlapping segments using tiktoken.

        Args:
            text: Text to chunk
            source: Source file path

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        tokens = self.tokenizer.encode(text)
        chunks = []

        chunk_size = self.rag_config.chunk_size
        overlap = self.rag_config.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Find character positions (approximate)
            start_char = len(self.tokenizer.decode(tokens[:start]))
            end_char = len(self.tokenizer.decode(tokens[:end]))

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=len(chunk_tokens),
                )
            )

            chunk_index += 1
            start += chunk_size - overlap

            # Prevent infinite loop
            if chunk_size <= overlap:
                break

        return chunks

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hex string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _embed_chunks(self, chunks: List[TextChunk]) -> List[PointStruct]:
        """Generate embeddings and create Qdrant points.

        Args:
            chunks: List of text chunks

        Returns:
            List of Qdrant points
        """
        texts = [c.text for c in chunks]
        embeddings = await self.embedding_service.embed_batch(texts)

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "token_count": chunk.token_count,
                },
            )
            points.append(point)

        return points
