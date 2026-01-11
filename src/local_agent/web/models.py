"""Pydantic models for web API requests and responses."""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    message: str
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    """Response after sending a chat message."""

    status: Literal["executing"]


class ExecutionStatusResponse(BaseModel):
    """Response with current execution status."""

    status: Literal["executing", "completed", "error"]
    response: str | None = None
    error: str | None = None


class ApprovalInfo(BaseModel):
    """Information about a pending approval request."""

    approval_id: str
    tool_name: str
    description: str
    risk_tier: str
    parameters: Dict[str, Any]
    created_at: str


class PendingApprovalsResponse(BaseModel):
    """Response with list of pending approvals."""

    approvals: List[ApprovalInfo]


class ApprovalDecisionRequest(BaseModel):
    """Request to approve or deny a tool execution."""

    reason: str | None = None


class ApprovalDecisionResponse(BaseModel):
    """Response after approval decision."""

    success: bool


class MessageInfo(BaseModel):
    """Information about a single message."""

    role: str
    content: str


class MessageHistoryResponse(BaseModel):
    """Response with conversation history."""

    messages: List[MessageInfo]
