"""SQLAlchemy models for threads and messages."""

from __future__ import annotations
import datetime as dt
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.orm import relationship
from .db import Base


def utcnow():
    """Get current UTC time (timezone-aware)."""
    return dt.datetime.now(dt.UTC)


class Thread(Base):
    """A conversation thread."""

    __tablename__ = "threads"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow)

    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")


class Message(Base):
    """A message in a thread."""

    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    thread_id = Column(String, ForeignKey("threads.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utcnow)

    thread = relationship("Thread", back_populates="messages")


class Session(Base):
    """An agent execution session.

    A session represents a single execution context (CLI or UI client).
    Multiple sessions can exist over time for the same thread.
    """

    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    thread_id = Column(String, ForeignKey("threads.id"), nullable=True)
    client_type = Column(String, nullable=False)  # 'cli' or 'ui'
    status = Column(String, nullable=False)  # 'active', 'closed', 'crashed'
    created_at = Column(DateTime, nullable=False, default=utcnow)
    updated_at = Column(DateTime, nullable=False, default=utcnow)
    ended_at = Column(DateTime, nullable=True)

    # Session metadata
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    total_turns = Column(Integer, default=0)
    total_tool_calls = Column(Integer, default=0)


class MessageMeta(Base):
    """Metadata for assistant messages.

    Stores LLM response metadata like token counts, latency, and tool calls.
    """

    __tablename__ = "message_meta"

    id = Column(String, primary_key=True)
    message_id = Column(String, ForeignKey("messages.id"), nullable=False)
    model = Column(String, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    tool_calls_json = Column(Text, nullable=True)  # JSON array of tool call info

    message = relationship("Message", backref="meta")
