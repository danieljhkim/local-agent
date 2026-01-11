"""FastAPI application for Local Agent web service."""

from __future__ import annotations
import os
import uuid
import asyncio
import datetime as dt

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import select, desc

from ..config.loader import load_config
from .dependencies import set_session_manager
from .session_manager import SessionManager
from .routes import sessions, chat, approvals
from ..persistence.db import engine, SessionLocal, Base
from ..persistence.db_models import Thread, Message


# Load configuration
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Local Agent API",
    description="Web API for Local Agent with tool execution and approval workflow",
    version="0.1.0",
)

# CORS configuration from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.web.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session manager (single-user local deployment)
session_manager = SessionManager(config)
set_session_manager(session_manager)


# Include routers
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(approvals.router)


# Startup: Start cleanup task
@app.on_event("startup")
async def startup():
    """Start background tasks on application startup."""
    # Create storage directory and initialize database
    os.makedirs("storage", exist_ok=True)
    Base.metadata.create_all(bind=engine)

    # Start session cleanup
    asyncio.create_task(session_manager.cleanup_inactive_sessions())


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Status dict
    """
    return {"status": "healthy"}


# ========================================
# Thread-based Chat API (Ollama + RAG)
# ========================================


class CreateThreadResponse(BaseModel):
    """Response for creating a new thread."""

    thread_id: str


@app.post("/threads", response_model=CreateThreadResponse)
def create_thread() -> CreateThreadResponse:
    """Create a new conversation thread.

    Returns:
        CreateThreadResponse with thread_id
    """
    tid = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(
            Thread(
                id=tid,
                title="New chat",
                created_at=dt.datetime.utcnow(),
                updated_at=dt.datetime.utcnow(),
            )
        )
        db.commit()
    return CreateThreadResponse(thread_id=tid)


class ChatRequest(BaseModel):
    """Request for thread-based chat."""

    thread_id: str
    message: str
    use_rag: bool = True


class ChatResponse(BaseModel):
    """Response from thread-based chat."""

    reply: str
    citations: list[dict]


# @app.post("/chat", response_model=ChatResponse)
# def chat_with_thread(req: ChatRequest) -> ChatResponse:
#     """Send a message in a thread and get a response.

#     Args:
#         req: Chat request with thread_id, message, and RAG flag

#     Returns:
#         ChatResponse with reply and citations

#     Raises:
#         HTTPException: If thread not found
#     """
#     with SessionLocal() as db:
#         thread = db.get(Thread, req.thread_id)
#         if thread is None:
#             raise HTTPException(status_code=404, detail="thread not found")

#         # Store user message
#         umsg = Message(
#             id=str(uuid.uuid4()),
#             thread_id=req.thread_id,
#             role="user",
#             content=req.message,
#         )
#         db.add(umsg)
#         thread.updated_at = dt.datetime.utcnow()
#         db.commit()

#         # Load last N messages for context
#         stmt = (
#             select(Message)
#             .where(Message.thread_id == req.thread_id)
#             .order_by(desc(Message.created_at))
#             .limit(20)
#         )
#         recent = list(reversed(db.execute(stmt).scalars().all()))
#         messages = [{"role": m.role, "content": m.content} for m in recent]

#         citations: list[dict] = []
#         if req.use_rag:
#             chunks = rag.retrieve(req.message)
#             if chunks:
#                 citations = [{"source": c.source, "score": c.score} for c in chunks]
#                 context_block = "\n\n".join(
#                     f"[Source: {c.source} | score={c.score:.3f}]\n{c.text}"
#                     for c in chunks
#                 )
#                 system = {
#                     "role": "system",
#                     "content": (
#                         "You are a helpful assistant. Use the provided CONTEXT when relevant. "
#                         "If CONTEXT is insufficient, say so explicitly.\n\n"
#                         f"CONTEXT:\n{context_block}"
#                     ),
#                 }
#                 messages = [system] + messages

#         reply = ollama.chat(messages)

#         # Store assistant message
#         amsg = Message(
#             id=str(uuid.uuid4()),
#             thread_id=req.thread_id,
#             role="assistant",
#             content=reply,
#         )
#         db.add(amsg)
#         thread.updated_at = dt.datetime.utcnow()
#         db.commit()

#     return ChatResponse(reply=reply, citations=citations)
