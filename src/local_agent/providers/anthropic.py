"""Anthropic (Claude) provider implementation."""

import os
from typing import Any, AsyncIterator, Dict, List

from anthropic import AsyncAnthropic

from .base import CompletionResponse, LLMProvider, Message, ToolCall


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
            model: Model name (default: claude-3-5-sonnet-20241022)
        """
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or "claude-3-5-sonnet-20241022"
        self.client = AsyncAnthropic(api_key=self.api_key)

    async def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using Anthropic API.

        Args:
            messages: Conversation history
            tools: Available tools
            **kwargs: Additional arguments

        Returns:
            CompletionResponse
        """
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                # Handle both string and structured content
                content = msg.content
                if isinstance(content, str):
                    content = content  # Keep as string
                # If content is a list (structured blocks), pass as-is
                anthropic_messages.append({"role": msg.role, "content": content})

        # Make API call
        request_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system_message:
            request_kwargs["system"] = system_message

        if tools:
            request_kwargs["tools"] = tools

        response = await self.client.messages.create(**request_kwargs)

        # Parse response
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id, name=block.name, parameters=block.input
                    )
                )

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
        )

    async def stream_complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResponse]:
        """Generate a streaming completion.

        Args:
            messages: Conversation history
            tools: Available tools
            **kwargs: Additional arguments

        Yields:
            CompletionResponse chunks
        """
        # Convert messages
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                # Handle both string and structured content
                content = msg.content
                if isinstance(content, str):
                    content = content  # Keep as string
                # If content is a list (structured blocks), pass as-is
                anthropic_messages.append({"role": msg.role, "content": content})

        # Make streaming API call
        request_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system_message:
            request_kwargs["system"] = system_message

        if tools:
            request_kwargs["tools"] = tools

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                # Handle different event types
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield CompletionResponse(content=event.delta.text)
                    elif event.type == "message_stop":
                        yield CompletionResponse(finish_reason="stop")
