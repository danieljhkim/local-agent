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

    class Config:
        env_prefix = "AGENT_"
        env_nested_delimiter = "__"
