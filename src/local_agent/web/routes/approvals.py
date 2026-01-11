"""Approval management routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_session_manager
from ..models import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalInfo,
    PendingApprovalsResponse,
)
from ..session_manager import SessionManager

router = APIRouter(prefix="/api/sessions/{session_id}/approvals", tags=["approvals"])


@router.get("/pending", response_model=PendingApprovalsResponse)
async def get_pending_approvals(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Get list of pending approval requests.

    Args:
        session_id: Session identifier

    Returns:
        PendingApprovalsResponse with list of pending approvals
    """
    session = session_manager.get_session(session_id)

    pending = session.approval_queue.get_pending_approvals()

    approvals = [
        ApprovalInfo(
            approval_id=p.approval_id,
            tool_name=p.tool_schema.name,
            description=p.tool_schema.description,
            risk_tier=p.tool_schema.risk_tier.value,
            parameters=p.parameters,
            created_at=p.created_at.isoformat(),
        )
        for p in pending
    ]

    return PendingApprovalsResponse(approvals=approvals)


@router.post("/{approval_id}/approve", response_model=ApprovalDecisionResponse)
async def approve_request(
    session_id: str,
    approval_id: str,
    request: ApprovalDecisionRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Approve a pending tool execution request.

    Args:
        session_id: Session identifier
        approval_id: Approval request identifier
        request: Approval decision with optional reason

    Returns:
        ApprovalDecisionResponse with success status

    Raises:
        HTTPException: If approval not found (404)
    """
    session = session_manager.get_session(session_id)

    try:
        await session.approval_queue.respond_to_approval(
            approval_id, approved=True, reason=request.reason
        )
        return ApprovalDecisionResponse(success=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{approval_id}/deny", response_model=ApprovalDecisionResponse)
async def deny_request(
    session_id: str,
    approval_id: str,
    request: ApprovalDecisionRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Deny a pending tool execution request.

    Args:
        session_id: Session identifier
        approval_id: Approval request identifier
        request: Approval decision with optional reason

    Returns:
        ApprovalDecisionResponse with success status

    Raises:
        HTTPException: If approval not found (404)
    """
    session = session_manager.get_session(session_id)

    try:
        await session.approval_queue.respond_to_approval(
            approval_id, approved=False, reason=request.reason
        )
        return ApprovalDecisionResponse(success=True)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
