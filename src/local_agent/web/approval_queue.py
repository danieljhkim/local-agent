"""Queue-based approval system for web interface."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from ..tools.schema import ToolSchema


class PendingApproval(BaseModel):
    """A pending approval request awaiting user response."""

    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_schema: ToolSchema
    parameters: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

    # Response state (not serialized to JSON)
    response_event: asyncio.Event = Field(
        default_factory=asyncio.Event, exclude=True, repr=False
    )
    approved: bool | None = None
    reason: str | None = None

    class Config:
        arbitrary_types_allowed = True  # Allow asyncio.Event


class ApprovalQueue:
    """Queue-based approval system for web interface.

    This replaces the blocking CLI approval workflow with an async system
    that allows web clients to approve/deny tool executions.
    """

    def __init__(self, timeout_seconds: int = 300):
        """Initialize approval queue.

        Args:
            timeout_seconds: Maximum time to wait for approval response (default: 5 minutes)
        """
        self.pending: Dict[str, PendingApproval] = {}
        self.timeout_seconds = timeout_seconds

    async def request_approval(
        self, tool_schema: ToolSchema, parameters: Dict[str, Any]
    ) -> Tuple[bool, str | None]:
        """Request approval for a tool execution and wait for user response.

        This method blocks until the user responds via the web UI or timeout occurs.

        Args:
            tool_schema: Schema of the tool requesting approval
            parameters: Parameters for the tool call

        Returns:
            Tuple of (approved: bool, reason: str | None)

        Raises:
            TimeoutError: If approval request times out
        """
        # Create pending approval with unique ID
        approval = PendingApproval(
            tool_schema=tool_schema,
            parameters=parameters,
        )

        self.pending[approval.approval_id] = approval

        try:
            # Wait for user response via API endpoint
            await asyncio.wait_for(
                approval.response_event.wait(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            # Clean up on timeout
            del self.pending[approval.approval_id]
            raise TimeoutError(
                f"Approval request timed out after {self.timeout_seconds}s"
            )

        # User responded - remove from pending and return decision
        del self.pending[approval.approval_id]
        return approval.approved, approval.reason

    def get_pending_approvals(self) -> List[PendingApproval]:
        """Get list of all pending approval requests.

        Returns:
            List of pending approvals
        """
        return list(self.pending.values())

    async def respond_to_approval(
        self, approval_id: str, approved: bool, reason: str | None = None
    ):
        """Respond to a pending approval request.

        Called by API endpoint when user approves or denies a tool execution.

        Args:
            approval_id: ID of the approval request
            approved: Whether the user approved the action
            reason: Optional reason for the decision

        Raises:
            ValueError: If approval ID not found or already responded
        """
        approval = self.pending.get(approval_id)
        if not approval:
            raise ValueError(f"Approval {approval_id} not found or already responded")

        # Set response
        approval.approved = approved
        approval.reason = reason or (
            "Approved by user" if approved else "Denied by user"
        )

        # Signal waiting request_approval() call
        approval.response_event.set()
