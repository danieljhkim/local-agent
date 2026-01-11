"""Tool registry for managing available tools."""

from typing import Any, Callable, Dict

from .schema import RiskTier, ToolSchema


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, ToolSchema] = {}

    def register(
        self,
        name: str,
        description: str,
        risk_tier: RiskTier,
        handler: Callable[..., Any],
        parameters: list = None,
    ) -> None:
        """Register a new tool.

        Args:
            name: Unique tool name
            description: Human-readable description
            risk_tier: Risk tier for approval workflow
            handler: Function that implements the tool
            parameters: List of ToolParameter objects
        """
        tool = ToolSchema(
            name=name,
            description=description,
            risk_tier=risk_tier,
            parameters=parameters or [],
            handler=handler,
        )
        self._tools[name] = tool

    def get(self, name: str) -> ToolSchema | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_tier(self, tier: RiskTier) -> list[ToolSchema]:
        """List tools by risk tier."""
        return [tool for tool in self._tools.values() if tool.risk_tier == tier]
