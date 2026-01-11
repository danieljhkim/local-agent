"""Dependency injection utilities for FastAPI."""

from .session_manager import SessionManager

# Global session manager instance (initialized in app.py)
_session_manager: SessionManager | None = None


def set_session_manager(manager: SessionManager):
    """Set the global session manager instance.

    Args:
        manager: SessionManager instance
    """
    global _session_manager
    _session_manager = manager


def get_session_manager() -> SessionManager:
    """Dependency function for session manager.

    Returns:
        Global session manager instance

    Raises:
        RuntimeError: If session manager not initialized
    """
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized")
    return _session_manager
