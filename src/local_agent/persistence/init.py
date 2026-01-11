"""Database initialization and migration utilities."""

from pathlib import Path
from typing import Optional

from .db import Base, engine, DATABASE_URL
from .db_models import Message, MessageMeta, Session, Thread


def init_database(db_path: Optional[Path] = None) -> None:
    """Initialize database tables.

    Creates all tables defined in db_models if they don't exist.
    Safe to call multiple times - will not drop existing tables.

    Args:
        db_path: Optional custom database path (currently unused, for future use)
    """
    # Create all tables
    Base.metadata.create_all(bind=engine)


def reset_database() -> None:
    """Reset database (DESTRUCTIVE - deletes all data!).

    Drops all tables and recreates them.
    Use with caution - this will delete all threads, messages, and sessions.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def get_database_info() -> dict:
    """Get database information.

    Returns:
        Dict with database metadata including path and table names
    """
    return {
        "database_url": DATABASE_URL,
        "tables": [table.name for table in Base.metadata.sorted_tables],
    }


def check_database_exists() -> bool:
    """Check if database file exists.

    Returns:
        True if database file exists, False otherwise
    """
    # Extract path from DATABASE_URL (format: sqlite:///path/to/db.db)
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        return db_path.exists()
    return False


def check_tables_exist() -> bool:
    """Check if all required tables exist in the database.

    Returns:
        True if all tables exist, False otherwise
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    required_tables = {"threads", "messages", "sessions", "message_meta"}

    return required_tables.issubset(existing_tables)
