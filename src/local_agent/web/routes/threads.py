"""Thread management routes."""

from typing import Optional
import uuid
import datetime as dt

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, func, desc

from ...persistence.db import SessionLocal
from ...persistence.db_models import Thread, Message
from ..models import (
    ThreadInfo,
    ListThreadsResponse,
    UpdateThreadRequest,
    CreateThreadRequest,
    CreateSessionResponse,
)

router = APIRouter(prefix="/api/threads", tags=["threads"])


@router.post("", response_model=CreateSessionResponse)
def create_thread(request: Optional[CreateThreadRequest] = None):
    """Create a new conversation thread.

    Args:
        request: Optional thread configuration with title

    Returns:
        CreateSessionResponse with thread_id
    """
    thread_id = str(uuid.uuid4())
    title = request.title if request and request.title else "New chat"

    with SessionLocal() as db:
        thread = Thread(
            id=thread_id,
            title=title,
            created_at=dt.datetime.now(dt.UTC),
            updated_at=dt.datetime.now(dt.UTC),
        )
        db.add(thread)
        db.commit()

    return CreateSessionResponse(session_id=thread_id)


@router.get("", response_model=ListThreadsResponse)
def list_threads(limit: int = 50, offset: int = 0):
    """List conversation threads ordered by most recent.

    Args:
        limit: Maximum number of threads to return (default: 50)
        offset: Number of threads to skip (default: 0)

    Returns:
        ListThreadsResponse with list of threads and message counts
    """
    with SessionLocal() as db:
        # Query threads with message counts
        stmt = (
            select(Thread, func.count(Message.id).label("message_count"))
            .outerjoin(Message, Thread.id == Message.thread_id)
            .group_by(Thread.id)
            .order_by(desc(Thread.updated_at))
            .limit(limit)
            .offset(offset)
        )

        results = db.execute(stmt).all()

        threads = [
            ThreadInfo(
                id=thread.id,
                title=thread.title,
                created_at=thread.created_at.isoformat(),
                updated_at=thread.updated_at.isoformat(),
                message_count=message_count,
            )
            for thread, message_count in results
        ]

    return ListThreadsResponse(threads=threads)


@router.get("/{thread_id}", response_model=ThreadInfo)
def get_thread(thread_id: str):
    """Get a single thread by ID.

    Args:
        thread_id: Thread identifier

    Returns:
        ThreadInfo with thread details

    Raises:
        HTTPException: If thread not found (404)
    """
    with SessionLocal() as db:
        thread = db.get(Thread, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Count messages
        message_count = (
            db.query(func.count(Message.id))
            .filter(Message.thread_id == thread_id)
            .scalar()
        )

        return ThreadInfo(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at.isoformat(),
            updated_at=thread.updated_at.isoformat(),
            message_count=message_count,
        )


@router.patch("/{thread_id}", response_model=ThreadInfo)
def update_thread(thread_id: str, request: UpdateThreadRequest):
    """Update thread metadata.

    Args:
        thread_id: Thread identifier
        request: Update request with fields to modify

    Returns:
        Updated ThreadInfo

    Raises:
        HTTPException: If thread not found (404)
    """
    with SessionLocal() as db:
        thread = db.get(Thread, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Update fields
        if request.title is not None:
            thread.title = request.title

        thread.updated_at = dt.datetime.now(dt.UTC)
        db.commit()
        db.refresh(thread)

        # Count messages
        message_count = (
            db.query(func.count(Message.id))
            .filter(Message.thread_id == thread_id)
            .scalar()
        )

        return ThreadInfo(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at.isoformat(),
            updated_at=thread.updated_at.isoformat(),
            message_count=message_count,
        )


@router.delete("/{thread_id}")
def delete_thread(thread_id: str):
    """Delete a thread and all its messages.

    Args:
        thread_id: Thread identifier

    Returns:
        Success status

    Raises:
        HTTPException: If thread not found (404)
    """
    with SessionLocal() as db:
        thread = db.get(Thread, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # SQLAlchemy cascade delete will handle messages
        db.delete(thread)
        db.commit()

    return {"success": True}
