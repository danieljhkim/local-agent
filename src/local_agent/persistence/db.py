"""Database setup and session management."""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Get database path from environment or use default
# This can be overridden by setting AGENT_STATE_DIR environment variable
DB_DIR = os.environ.get("AGENT_STATE_DIR", str(Path.home() / ".local" / "agent" / "state"))
DB_PATH = Path(DB_DIR) / "local_agent.db"

# Ensure directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite database URL
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
