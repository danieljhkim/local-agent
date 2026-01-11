"""Tool schema definitions using Pydantic."""

from enum import Enum
from typing import Any, Callable, Dict

from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    """Risk tier for tools determining approval requirements."""

    TIER_0 = "tier_0"  # Read-only operations
    TIER_1 = "tier_1"  # Drafting/proposed changes (no side effects)
    TIER_2 = "tier_2"  # Side-effectful operations (requires approval)


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class ToolSchema(BaseModel):
    """Schema for a tool definition."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Human-readable tool description")
    risk_tier: RiskTier = Field(..., description="Risk tier for approval workflow")
    parameters: list[ToolParameter] = Field(
        default_factory=list, description="Tool parameters"
    )
    handler: Callable[..., Any] | None = Field(
        default=None, description="Tool handler function", exclude=True
    )

    class Config:
        arbitrary_types_allowed = True


class ToolResult(BaseModel):
    """Result of a tool execution."""

    success: bool
    result: Any = None
    error: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
