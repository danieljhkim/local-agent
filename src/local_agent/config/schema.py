"""Configuration schema for the agent."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class WorkspaceConfig(BaseModel):
    """Workspace configuration for filesystem access."""

    allowed_roots: list[str] = Field(
        default_factory=list,
        description="List of allowed root directories for filesystem access",
    )
    denied_paths: list[str] = Field(
        default_factory=lambda: [
            "~/.ssh",
            "~/.aws",
            "~/Library/Keychains",
            "/etc",
            "/System",
        ],
        description="List of explicitly denied paths",
    )


class ApprovalPolicy(BaseModel):
    """Approval policy for tool execution."""

    tool_pattern: str = Field(..., description="Tool name or pattern (e.g., 'fs_*')")
    auto_approve: bool = Field(
        default=False, description="Whether to auto-approve matching tools"
    )
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="Conditions for auto-approval"
    )


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    name: str = Field(..., description="Provider name (anthropic, openai, etc.)")
    api_key: str | None = Field(
        default=None, description="API key (if not using env var)"
    )
    model: str = Field(..., description="Model name to use")
    base_url: str | None = Field(default=None, description="Optional custom base URL")


class AuditConfig(BaseModel):
    """Audit logging configuration."""

    enabled: bool = Field(default=True, description="Enable audit logging")
    log_dir: str = Field(
        default="~/.local/agent/logs", description="Directory for audit logs"
    )
    redact_patterns: list[str] = Field(
        default_factory=lambda: [
            r"api[_-]?key",
            r"token",
            r"password",
            r"secret",
            r"-----BEGIN.*PRIVATE KEY-----",
        ],
        description="Patterns to redact from logs",
    )


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    collection_name: str = Field(
        default="local_agent_docs",
        description="Collection name for document embeddings",
    )
    vector_size: int = Field(
        default=768, description="Embedding dimensions (768 for nomic-embed-text)"
    )
    timeout: float = Field(default=30.0, description="Request timeout in seconds")


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""

    model: str = Field(
        default="nomic-embed-text:latest", description="Ollama embedding model"
    )
    base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    batch_size: int = Field(
        default=32, description="Batch size for embedding generation"
    )
    timeout: float = Field(default=120.0, description="Request timeout in seconds")


class RAGConfig(BaseModel):
    """RAG pipeline configuration."""

    chunk_size: int = Field(
        default=512, description="Target chunk size in tokens (400-800 recommended)"
    )
    chunk_overlap: int = Field(
        default=128, description="Overlap between chunks in tokens"
    )
    top_k: int = Field(default=5, description="Default number of retrieval results")
    score_threshold: float = Field(
        default=0.0, description="Minimum similarity score (0.0-1.0)"
    )
    supported_extensions: list[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
        ],
        description="Supported file extensions",
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = Field(
        default="~/.local/agent/state/local_agent.db",
        description="SQLite database file path",
    )


class WebConfig(BaseModel):
    """Web server configuration."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"],
        description="Allowed CORS origins",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to listen on")


class AgentConfig(BaseSettings):
    """Main agent configuration."""

    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    providers: list[ProviderConfig] = Field(
        default_factory=list, description="Available LLM providers"
    )
    default_provider: str | None = Field(
        default=None, description="Default provider to use"
    )
    approval_policies: list[ApprovalPolicy] = Field(
        default_factory=list, description="Approval policies for tools"
    )
    audit: AuditConfig = Field(default_factory=AuditConfig)
    state_dir: str = Field(
        default="~/.local/agent/state", description="Directory for state storage"
    )
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    class Config:
        env_prefix = "AGENT_"
        env_nested_delimiter = "__"
