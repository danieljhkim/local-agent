"""Web-based approval workflow using queue system."""

from typing import Any, Dict, Tuple

from ..tools.schema import ToolSchema
from .approval_queue import ApprovalQueue


class WebApprovalWorkflow:
    """Web-based approval workflow using async queue system.

    This class provides a drop-in replacement for the CLI ApprovalWorkflow,
    delegating approval requests to an ApprovalQueue that web clients can
    interact with via REST API.
    """

    def __init__(self, approval_queue: ApprovalQueue):
        """Initialize web approval workflow.

        Args:
            approval_queue: Approval queue to use for handling requests
        """
        self.queue = approval_queue

    async def request_approval(
        self, tool: ToolSchema, parameters: Dict[str, Any]
    ) -> Tuple[bool, str | None]:
        """Request approval for a tool execution via web queue.

        This is an async method (unlike the CLI version) that delegates to
        the approval queue and blocks until the user responds via the web UI.

        Args:
            tool: Tool schema
            parameters: Tool parameters

        Returns:
            Tuple of (approved, reason)

        Raises:
            TimeoutError: If approval request times out
        """
        return await self.queue.request_approval(tool, parameters)
