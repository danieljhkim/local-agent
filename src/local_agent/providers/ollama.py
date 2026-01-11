"""Ollama provider implementation for local LLM inference."""

import json
import os
from typing import Any, AsyncIterator, Dict, List

import httpx

from .base import CompletionResponse, LLMProvider, Message, ToolCall


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize Ollama provider.

        Args:
            api_key: Not used for Ollama (local deployment)
            model: Model name (default: from OLLAMA_CHAT_MODEL or llama3.1:8b)
            base_url: Ollama server URL (default: from OLLAMA_BASE_URL or localhost:11434)
        """
        super().__init__(api_key, model)
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        self.model = model or os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
        timeout_str = os.getenv("OLLAMA_TIMEOUT", "120")
        self.timeout = float(timeout_str)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a completion using Ollama API.

        Args:
            messages: Conversation history
            tools: Available tools (gracefully ignored if model doesn't support)
            **kwargs: Additional arguments

        Returns:
            CompletionResponse
        """
        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                content = msg.content
                # Handle structured content (from Anthropic-style messages)
                if isinstance(content, list):
                    # Convert structured blocks to text
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                # Format tool results as text
                                text_parts.append(f"Tool result: {block.get('content', '')}")
                    content = "\n".join(text_parts)
                ollama_messages.append({"role": msg.role, "content": content})

            # Build request
            request_data = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
            }

            # Include tools if provided (gracefully handled by Ollama)
            if tools:
                request_data["tools"] = tools

            # Make API call
            url = f"{self.base_url}/api/chat"
            client = self._get_client()
            response = await client.post(url, json=request_data)
            response.raise_for_status()
            data = response.json()

            # Parse response
            message = data.get("message", {})
            content = message.get("content", "")
            tool_calls = []

            # Handle tool calls if present (graceful - may not exist)
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("function", {}).get("name", ""),
                            parameters=tc.get("function", {}).get("arguments", {}),
                        )
                    )

            return CompletionResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=data.get("done_reason", "stop"),
            )

        except httpx.ConnectError:
            return CompletionResponse(
                content="Error: Cannot connect to Ollama. Is it running at {self.base_url}?",
                finish_reason="error",
            )
        except httpx.TimeoutException:
            return CompletionResponse(
                content=f"Error: Ollama request timed out after {self.timeout}s",
                finish_reason="error",
            )
        except Exception as e:
            return CompletionResponse(
                content=f"Error calling Ollama: {str(e)}",
                finish_reason="error",
            )

    async def stream_complete(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[CompletionResponse]:
        """Generate a streaming completion using Ollama API.

        Args:
            messages: Conversation history
            tools: Available tools (gracefully ignored if model doesn't support)
            **kwargs: Additional arguments

        Yields:
            CompletionResponse chunks
        """
        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                content = msg.content
                # Handle structured content (from Anthropic-style messages)
                if isinstance(content, list):
                    # Convert structured blocks to text
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                # Format tool results as text
                                text_parts.append(f"Tool result: {block.get('content', '')}")
                    content = "\n".join(text_parts)
                ollama_messages.append({"role": msg.role, "content": content})

            # Build request
            request_data = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
            }

            # Include tools if provided
            if tools:
                request_data["tools"] = tools

            # Make streaming API call
            url = f"{self.base_url}/api/chat"
            client = self._get_client()
            async with client.stream("POST", url, json=request_data) as response:
                response.raise_for_status()

                # Process NDJSON stream
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                        message = chunk.get("message", {})
                        content = message.get("content", "")

                        # Yield content if present
                        if content:
                            yield CompletionResponse(content=content)

                        # Handle final chunk
                        if chunk.get("done", False):
                            # Parse tool calls if present
                            tool_calls = []
                            if "tool_calls" in message and message["tool_calls"]:
                                for tc in message["tool_calls"]:
                                    tool_calls.append(
                                        ToolCall(
                                            id=tc.get("id", ""),
                                            name=tc.get("function", {}).get("name", ""),
                                            parameters=tc.get("function", {}).get(
                                                "arguments", {}
                                            ),
                                        )
                                    )

                            yield CompletionResponse(
                                tool_calls=tool_calls,
                                finish_reason=chunk.get("done_reason", "stop"),
                            )

                    except json.JSONDecodeError:
                        # Skip invalid JSON lines
                        continue

        except httpx.ConnectError:
            yield CompletionResponse(
                content=f"Error: Cannot connect to Ollama. Is it running at {self.base_url}?",
                finish_reason="error",
            )
        except httpx.TimeoutException:
            yield CompletionResponse(
                content=f"Error: Ollama request timed out after {self.timeout}s",
                finish_reason="error",
            )
        except Exception as e:
            yield CompletionResponse(
                content=f"Error calling Ollama: {str(e)}",
                finish_reason="error",
            )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()