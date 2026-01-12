"""Database setup and session management."""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def get_database_path() -> Path:
    """Get database path from environment or default.

    Priority:
    1. AGENT_DATABASE_PATH env var (explicit path)
    2. AGENT_STATE_DIR env var + local_agent.db
    3. Default: ~/.local/share/local-agent/state/local_agent.db
    """
    # Check for explicit database path
    if db_path := os.environ.get("AGENT_DATABASE_PATH"):
        return Path(db_path).expanduser()

    # Check for state directory
    state_dir = os.environ.get(
        "AGENT_STATE_DIR",
        str(Path.home() / ".local" / "share" / "local-agent" / "state"),
    )
    return Path(state_dir) / "local_agent.db"


# Get database path
DB_PATH = get_database_path()

# Ensure directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite database URL
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
