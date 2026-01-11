"""OpenAI provider implementation."""

import os
from typing import Any, AsyncIterator, Dict, List

from openai import AsyncOpenAI

from .base import CompletionResponse, LLMProvider, Message, ToolCall


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: Model name (default: gpt-4-turbo-preview)
        """
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or "gpt-4-turbo-preview"
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using OpenAI API.

        Args:
            messages: Conversation history
            tools: Available tools
            **kwargs: Additional arguments

        Returns:
            CompletionResponse
        """
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Make API call
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
        }

        if tools:
            request_kwargs["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]

        response = await self.client.chat.completions.create(**request_kwargs)

        # Parse response
        message = response.choices[0].message
        content = message.content
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        parameters=tc.function.arguments,
                    )
                )

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
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
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Make streaming API call
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }

        if tools:
            request_kwargs["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]

        stream = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta

                content = None
                if hasattr(delta, "content") and delta.content:
                    content = delta.content

                finish_reason = chunk.choices[0].finish_reason

                yield CompletionResponse(content=content, finish_reason=finish_reason)
