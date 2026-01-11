"""Chat execution routes."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_session_manager
from ..models import (
    ChatRequest,
    ChatResponse,
    ExecutionStatusResponse,
    MessageHistoryResponse,
)
from ..session_manager import SessionManager

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def send_message(
    session_id: str,
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Send a message and start agent execution.

    Starts execution in a background task and returns immediately.
    Client should poll /status to check completion.

    Args:
        session_id: Session identifier
        request: Chat request with message and optional system prompt

    Returns:
        ChatResponse with executing status

    Raises:
        HTTPException: If session already executing (409)
    """
    session = session_manager.get_session(session_id)

    # Check if already executing
    if session.execution_task and not session.execution_task.done():
        raise HTTPException(status_code=409, detail="Session already executing")

    # Reset state
    session.execution_complete.clear()
    session.final_response = None
    session.execution_error = None

    # Start execution in background
    async def execute_task():
        try:
            response = await session.runtime.execute(
                request.message, system_prompt=request.system_prompt
            )
            session.final_response = response
        except Exception as e:
            session.execution_error = e
        finally:
            session.execution_complete.set()

    session.execution_task = asyncio.create_task(execute_task())

    return ChatResponse(status="executing")


@router.get("/status", response_model=ExecutionStatusResponse)
async def get_execution_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Get current execution status.

    Args:
        session_id: Session identifier

    Returns:
        ExecutionStatusResponse with status, response, or error
    """
    session = session_manager.get_session(session_id)

    # Check if execution complete
    if session.execution_complete.is_set():
        if session.execution_error:
            return ExecutionStatusResponse(
                status="error", error=str(session.execution_error)
            )
        else:
            return ExecutionStatusResponse(
                status="completed", response=session.final_response
            )
    else:
        return ExecutionStatusResponse(status="executing")


@router.get("/history", response_model=MessageHistoryResponse)
async def get_message_history(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Get conversation history.

    Args:
        session_id: Session identifier

    Returns:
        MessageHistoryResponse with message list
    """
    session = session_manager.get_session(session_id)

    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in session.runtime.message_history
    ]

    return MessageHistoryResponse(messages=messages)
