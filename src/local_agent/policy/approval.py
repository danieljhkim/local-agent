"""Approval workflow for tool execution."""

from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

from ..tools.schema import ToolSchema


class ApprovalWorkflow:
    """Handles user approval for tool execution."""

    def __init__(self):
        self.console = Console()

    def request_approval(
        self, tool: ToolSchema, parameters: Dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Request user approval for a tool call.

        Args:
            tool: Tool schema
            parameters: Tool parameters

        Returns:
            Tuple of (approved, reason)
        """
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold yellow]Approval Required[/bold yellow]\n\n"
                f"[bold]Tool:[/bold] {tool.name}\n"
                f"[bold]Description:[/bold] {tool.description}\n"
                f"[bold]Risk Tier:[/bold] {tool.risk_tier.value}",
                title="🔒 Tool Execution Request",
                border_style="yellow",
            )
        )

        # Display parameters
        self.console.print("\n[bold]Parameters:[/bold]")
        for key, value in parameters.items():
            # Format value nicely
            if isinstance(value, str) and "\n" in value:
                # Multi-line string, show as syntax
                self.console.print(f"  [cyan]{key}:[/cyan]")
                syntax = Syntax(value, "text", theme="monokai", line_numbers=False)
                self.console.print(syntax)
            else:
                self.console.print(f"  [cyan]{key}:[/cyan] {value}")

        self.console.print()

        # Ask for approval
        approved = Confirm.ask(
            "[bold]Do you approve this action?[/bold]", default=False
        )

        reason = "Approved by user" if approved else "Denied by user"
        return approved, reason
