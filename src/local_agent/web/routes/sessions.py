"""Session management routes."""

from fastapi import APIRouter, Depends

from ..dependencies import get_session_manager
from ..models import CreateSessionResponse
from ..session_manager import SessionManager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Create a new agent session.

    Returns:
        CreateSessionResponse with session_id
    """
    session = await session_manager.create_session()
    return CreateSessionResponse(session_id=session.session_id)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a session and clean up resources.

    Args:
        session_id: Session identifier

    Returns:
        Success status
    """
    await session_manager.cleanup_session(session_id)
    return {"success": True}
