"""Session management for multi-user web service."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict

from fastapi import HTTPException

from ..config.schema import AgentConfig
from ..runtime.agent import AgentRuntime
from .approval_queue import ApprovalQueue
from .approval_workflow import WebApprovalWorkflow


class SessionInfo:
    """Information about an active agent session."""

    def __init__(
        self,
        session_id: str,
        runtime: AgentRuntime,
        approval_queue: ApprovalQueue,
        created_at: datetime,
    ):
        """Initialize session info.

        Args:
            session_id: Unique session identifier
            runtime: Agent runtime instance
            approval_queue: Approval queue for this session
            created_at: Session creation timestamp
        """
        self.session_id = session_id
        self.runtime = runtime
        self.approval_queue = approval_queue
        self.created_at = created_at
        self.last_activity = created_at

        # Execution state
        self.execution_task: asyncio.Task | None = None
        self.execution_complete = asyncio.Event()
        self.final_response: str | None = None
        self.execution_error: Exception | None = None

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class SessionManager:
    """Manages multiple agent sessions for web service.

    Handles session creation, retrieval, cleanup, and timeout management.
    """

    def __init__(self, config: AgentConfig):
        """Initialize session manager.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.sessions: Dict[str, SessionInfo] = {}
        self.cleanup_interval_seconds = 300  # 5 minutes
        self.session_timeout_seconds = 3600  # 1 hour

    async def create_session(self) -> SessionInfo:
        """Create a new agent session.

        Returns:
            SessionInfo for the new session
        """
        session_id = str(uuid.uuid4())

        # Create approval queue for this session
        approval_queue = ApprovalQueue()

        # Create web approval workflow
        approval_workflow = WebApprovalWorkflow(approval_queue)

        # Create AgentRuntime with custom approval workflow
        runtime = AgentRuntime(self.config, approval_workflow=approval_workflow)

        session_info = SessionInfo(
            session_id=session_id,
            runtime=runtime,
            approval_queue=approval_queue,
            created_at=datetime.now(),
        )

        self.sessions[session_id] = session_info
        return session_info

    def get_session(self, session_id: str) -> SessionInfo:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo

        Raises:
            HTTPException: If session not found (404)
        """
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.update_activity()
        return session

    async def cleanup_session(self, session_id: str):
        """Clean up a session and its resources.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Cancel execution task if running
            if session.execution_task and not session.execution_task.done():
                session.execution_task.cancel()
                try:
                    await session.execution_task
                except asyncio.CancelledError:
                    pass

            del self.sessions[session_id]

    async def cleanup_inactive_sessions(self):
        """Background task to clean up inactive sessions.

        Runs indefinitely, checking for inactive sessions every cleanup_interval_seconds.
        Sessions inactive for longer than session_timeout_seconds are cleaned up.
        """
        while True:
            await asyncio.sleep(self.cleanup_interval_seconds)

            now = datetime.now()
            to_cleanup = []

            for session_id, session in self.sessions.items():
                inactive_duration = (now - session.last_activity).total_seconds()
                if inactive_duration > self.session_timeout_seconds:
                    to_cleanup.append(session_id)

            for session_id in to_cleanup:
                await self.cleanup_session(session_id)
