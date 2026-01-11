"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List

from pydantic import BaseModel


class Message(BaseModel):
    """Chat message."""

    role: str  # "user", "assistant", "system"
    content: str


class ToolCall(BaseModel):
    """Tool call from LLM."""

    id: str
    name: str
    parameters: Dict[str, Any]


class CompletionResponse(BaseModel):
    """Response from LLM completion."""

    content: str | None = None
    tool_calls: List[ToolCall] = []
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize provider.

        Args:
            api_key: API key for the provider
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion.

        Args:
            messages: Conversation history
            tools: Available tools (optional)
            **kwargs: Additional provider-specific arguments

        Returns:
            CompletionResponse
        """
        pass

    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResponse]:
        """Generate a streaming completion.

        Args:
            messages: Conversation history
            tools: Available tools (optional)
            **kwargs: Additional provider-specific arguments

        Yields:
            CompletionResponse chunks
        """
        pass
