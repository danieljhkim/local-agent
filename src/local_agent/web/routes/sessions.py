"""Session management routes."""

from typing import Optional

from fastapi import APIRouter, Depends

from ..dependencies import get_session_manager
from ..models import CreateSessionRequest, CreateSessionResponse
from ..session_manager import SessionManager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    request: Optional[CreateSessionRequest] = None,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Create a new agent session.

    Args:
        request: Optional session configuration with identity or system_prompt

    Returns:
        CreateSessionResponse with session_id
    """
    # Resolve system prompt from identity if provided
    system_prompt = None
    if request:
        if request.identity:
            from ...identities import get_identity_manager

            manager = get_identity_manager()
            try:
                system_prompt = manager.get_content(request.identity)
            except ValueError:
                # Fall back to active identity if specified identity not found
                pass
        elif request.system_prompt:
            system_prompt = request.system_prompt

    session = await session_manager.create_session(system_prompt=system_prompt)
    return CreateSessionResponse(session_id=session.session_id)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str, session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a session and clean up resources.

    Args:
        session_id: Session identifier

    Returns:
        Success status
    """
    await session_manager.cleanup_session(session_id)
    return {"success": True}
