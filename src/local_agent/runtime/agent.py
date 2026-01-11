"""Agent runtime orchestration."""

import asyncio
import functools
import os
import time
import traceback
import uuid
from typing import Any, Dict, List, Tuple

from rich.console import Console

from local_agent.providers.ollama import OllamaProvider

from ..audit.logger import AuditLogger
from ..config.schema import AgentConfig
from ..connectors.filesystem import FilesystemConnector
from ..policy.approval import ApprovalWorkflow
from ..policy.engine import PolicyEngine
from ..providers.anthropic import AnthropicProvider
from ..providers.base import CompletionResponse, LLMProvider, Message, ToolCall
from ..providers.openai import OpenAIProvider
from ..tools.filesystem import register_filesystem_tools
from ..tools.registry import ToolRegistry
from ..tools.schema import ToolParameter, ToolResult, ToolSchema

# Default system prompt to guide LLM behavior
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools for filesystem operations.

Guidelines:
- For simple questions, greetings, or conversations, respond directly without using tools
- Only use tools when you need to perform actual filesystem operations (read, write, search files)
- Read files before editing them to understand context
- Propose changes clearly before writing files
- If a tool fails, try alternative approaches or report the issue to the user
- Always verify critical operations completed successfully

Available tools will be provided in the tool schema."""


class AgentRuntime:
    """Agent runtime orchestration engine."""

    def __init__(self, config: AgentConfig, approval_workflow=None):
        """Initialize agent runtime.

        Args:
            config: Agent configuration
            approval_workflow: Optional approval workflow (defaults to CLI ApprovalWorkflow)
        """
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.message_history: List[Message] = []
        self.turn_count = 0
        self.max_turns = 20
        self.total_tool_calls = 0
        self.session_start_time = time.time()

        # Initialize components
        self.provider = self._init_provider()
        self.registry = self._init_registry()
        self.policy_engine = PolicyEngine(config)

        # Use provided approval workflow or default to CLI version
        if approval_workflow is None:
            approval_workflow = ApprovalWorkflow()
        self.approval_workflow = approval_workflow

        self.audit_logger = AuditLogger(config.audit, self.session_id)
        self.console = Console()

        # Log session start
        self.audit_logger.log_event(
            "session_start",
            {
                "session_id": self.session_id,
                "provider": config.default_provider,
                "max_turns": self.max_turns,
            },
        )

    def _init_provider(self) -> LLMProvider:
        """Initialize LLM provider based on config.

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider not found or not configured
        """
        provider_name = self.config.default_provider

        # Find provider config
        provider_config = None
        for p in self.config.providers:
            if p.name == provider_name:
                provider_config = p
                break

        if not provider_config:
            raise ValueError(f"Provider '{provider_name}' not found in config")

        # Get API key from config or environment
        api_key = provider_config.api_key
        if not api_key:
            # Try environment variable
            if provider_name == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif provider_name == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif provider_name == "ollama":
                api_key = "default"  # Ollama does not require an API key

        if not api_key:
            raise ValueError(
                f"API key not found for provider '{provider_name}'. "
                f"Set in config or environment variable."
            )

        # Instantiate provider
        if provider_name == "anthropic":
            return AnthropicProvider(api_key=api_key, model=provider_config.model)
        elif provider_name == "openai":
            return OpenAIProvider(api_key=api_key, model=provider_config.model)
        elif provider_name == "ollama":
            return OllamaProvider(
                api_key=api_key,
                model=provider_config.model,
                base_url=provider_config.base_url,
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def _init_registry(self) -> ToolRegistry:
        """Initialize tool registry and register tools.

        Returns:
            Tool registry with filesystem tools registered
        """
        registry = ToolRegistry()

        # Create filesystem connector with workspace config
        fs_connector = FilesystemConnector(self.config.workspace)

        # Register filesystem tools
        register_filesystem_tools(registry, fs_connector)

        return registry

    def _tools_to_llm_format(self) -> List[Dict[str, Any]]:
        """Convert tool schemas to LLM-compatible format.

        Returns:
            List of tool definitions in Anthropic-compatible format
        """
        tools = []

        for tool_schema in self.registry.list_tools():
            properties = {}
            required = []

            for param in tool_schema.parameters:
                param_def = {
                    "type": self._map_type(param.type),
                    "description": param.description,
                }

                if param.default is not None:
                    param_def["default"] = param.default

                properties[param.name] = param_def

                if param.required:
                    required.append(param.name)

            tools.append(
                {
                    "name": tool_schema.name,
                    "description": tool_schema.description,
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )

        return tools

    def _map_type(self, tool_type: str) -> str:
        """Map tool parameter type to JSON schema type.

        Args:
            tool_type: Tool parameter type

        Returns:
            JSON schema type
        """
        type_mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        return type_mapping.get(tool_type.lower(), "string")

    def _validate_and_coerce_parameters(
        self, tool_schema: ToolSchema, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and coerce parameters to expected types.

        Args:
            tool_schema: Tool schema
            parameters: Parameters from LLM

        Returns:
            Coerced parameters

        Raises:
            ValueError: If required parameter missing or invalid type
        """
        coerced = {}

        for param in tool_schema.parameters:
            value = parameters.get(param.name)

            # Handle missing parameters
            if value is None:
                if param.required and param.default is None:
                    raise ValueError(f"Missing required parameter: {param.name}")
                value = param.default

            # Type coercion
            if value is not None:
                if param.type == "integer" and isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        raise ValueError(
                            f"Parameter {param.name} must be an integer, got: {value}"
                        )
                elif param.type == "number" and isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(
                            f"Parameter {param.name} must be a number, got: {value}"
                        )
                elif param.type == "boolean" and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")

            coerced[param.name] = value

        return coerced

    def _format_tool_results(
        self, results: List[Tuple[ToolCall, ToolResult]]
    ) -> str:
        """Format tool execution results for LLM consumption.

        Args:
            results: List of (ToolCall, ToolResult) tuples

        Returns:
            Formatted text representation of results
        """
        formatted = "Tool Execution Results:\n\n"

        for tool_call, result in results:
            formatted += f"Tool: {tool_call.name}\n"
            formatted += f"Call ID: {tool_call.id}\n"

            if result.success:
                formatted += "Status: Success\n"
                result_str = self._truncate_result(result.result)
                formatted += f"Result: {result_str}\n"
                if result.metadata:
                    formatted += f"Metadata: {result.metadata}\n"
            else:
                formatted += "Status: Failed\n"
                formatted += f"Error: {result.error}\n"

            formatted += "\n"

        return formatted

    def _truncate_result(self, result_data: Any, max_chars: int = 2000) -> str:
        """Truncate large results for message history.

        Args:
            result_data: Result data to truncate
            max_chars: Maximum characters to include

        Returns:
            Truncated string representation
        """
        result_str = str(result_data)
        if len(result_str) > max_chars:
            return (
                result_str[:max_chars]
                + f"\n... (truncated, {len(result_str)} total chars)"
            )
        return result_str

    async def execute(
        self, user_message: str, system_prompt: str | None = None
    ) -> str:
        """Execute agent task with agentic loop.

        Args:
            user_message: User's message/task
            system_prompt: Optional system prompt (uses default if None)

        Returns:
            Final assistant response

        Raises:
            RuntimeError: If provider error or max turns exceeded
        """
        # Initialize message history for this execution
        if not self.message_history:
            # Only add system prompt on first message
            if system_prompt is None:
                system_prompt = DEFAULT_SYSTEM_PROMPT
            self.message_history.append(Message(role="system", content=system_prompt))

        # Add user message
        self.message_history.append(Message(role="user", content=user_message))

        # Get tool definitions
        tools = self._tools_to_llm_format()

        # Agentic loop
        final_response = None
        while self.turn_count < self.max_turns:
            self.turn_count += 1

            # Display thinking status
            with self.console.status(
                f"[cyan]Turn {self.turn_count}: Thinking...[/cyan]", spinner="dots"
            ):
                try:
                    response = await self.provider.complete(
                        self.message_history, tools=tools
                    )
                except Exception as e:
                    self.audit_logger.log_event(
                        "provider_error", {"error": str(e), "turn": self.turn_count}
                    )
                    raise RuntimeError(f"LLM provider error: {str(e)}") from e

            # Display and store assistant response
            if response.content:
                self.console.print(f"\n[bold green]Assistant:[/bold green] {response.content}")
                final_response = response.content

            # Check if done (no tool calls)
            if not response.tool_calls:
                # Add assistant message to history
                if response.content:
                    self.message_history.append(
                        Message(role="assistant", content=response.content)
                    )
                break

            # Process tool calls
            tool_results: List[Tuple[ToolCall, ToolResult]] = []

            for tool_call in response.tool_calls:
                self.total_tool_calls += 1

                # Display tool execution
                self.console.print(
                    f"\n[yellow]→[/yellow] Executing: [cyan]{tool_call.name}[/cyan]"
                )

                # Show parameters (truncated for display)
                params_str = ", ".join(
                    f"{k}={repr(v)[:50]}" for k, v in tool_call.parameters.items()
                )
                self.console.print(f"  [dim]Parameters: {params_str}[/dim]")

                # Get tool from registry
                tool_schema = self.registry.get(tool_call.name)
                if not tool_schema:
                    error_result = ToolResult(
                        success=False,
                        error=f"Tool '{tool_call.name}' not found in registry",
                    )
                    tool_results.append((tool_call, error_result))
                    self.console.print(
                        f"[red]✗ {tool_call.name} failed:[/red] Tool not found"
                    )
                    continue

                # Validate and coerce parameters
                try:
                    validated_params = self._validate_and_coerce_parameters(
                        tool_schema, tool_call.parameters
                    )
                except ValueError as e:
                    error_result = ToolResult(success=False, error=str(e))
                    tool_results.append((tool_call, error_result))
                    self.console.print(
                        f"[red]✗ {tool_call.name} failed:[/red] {str(e)}"
                    )
                    continue

                # Evaluate policy
                decision = self.policy_engine.evaluate(tool_schema, validated_params)

                if not decision.allowed:
                    error_result = ToolResult(
                        success=False, error=f"Policy denied: {decision.reason}"
                    )
                    tool_results.append((tool_call, error_result))
                    self.console.print(
                        f"[red]✗ {tool_call.name} denied:[/red] {decision.reason}"
                    )
                    continue

                # Request approval if needed
                if decision.requires_approval:
                    # Handle both sync (CLI) and async (web) approval workflows
                    if asyncio.iscoroutinefunction(self.approval_workflow.request_approval):
                        approved, reason = await self.approval_workflow.request_approval(
                            tool_schema, validated_params
                        )
                    else:
                        approved, reason = self.approval_workflow.request_approval(
                            tool_schema, validated_params
                        )

                    self.audit_logger.log_approval(tool_call.name, approved, reason)

                    if not approved:
                        error_result = ToolResult(
                            success=False,
                            error=f"User denied approval: {reason or 'No reason given'}",
                        )
                        tool_results.append((tool_call, error_result))
                        self.console.print("[red]✗ Approval denied[/red]\n")
                        continue

                    self.console.print("[green]✓ Approved[/green]\n")

                # Execute tool
                start_time = time.time()
                try:
                    # Handle both sync and async handlers
                    if asyncio.iscoroutinefunction(tool_schema.handler):
                        result = await tool_schema.handler(**validated_params)
                    else:
                        # Run sync handler in executor
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, functools.partial(tool_schema.handler, **validated_params)
                        )
                except Exception as e:
                    # Log the error
                    self.audit_logger.log_event(
                        "tool_execution_error",
                        {
                            "tool_name": tool_call.name,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )

                    result = ToolResult(
                        success=False,
                        error=f"Tool execution failed: {str(e)}",
                        metadata={"exception_type": type(e).__name__},
                    )

                elapsed_ms = (time.time() - start_time) * 1000

                # Log tool execution
                self.audit_logger.log_tool_call(
                    tool_name=tool_call.name,
                    risk_tier=tool_schema.risk_tier,
                    parameters=validated_params,
                    success=result.success,
                    result=result.result if result.success else None,
                    error=result.error,
                    elapsed_ms=elapsed_ms,
                )

                # Display result
                if result.success:
                    self.console.print(f"[green]✓ {tool_call.name} succeeded[/green]")
                    if result.metadata:
                        meta_str = ", ".join(
                            f"{k}={v}" for k, v in result.metadata.items()
                        )
                        self.console.print(f"  [dim]{meta_str}[/dim]")
                else:
                    self.console.print(
                        f"[red]✗ {tool_call.name} failed:[/red] {result.error}"
                    )

                tool_results.append((tool_call, result))

            # Add assistant message with tool calls to history
            # (We'll add a simplified text representation)
            if response.content:
                self.message_history.append(
                    Message(role="assistant", content=response.content)
                )
            else:
                # If no content, create a message indicating tool calls were made
                tool_calls_summary = ", ".join(tc.name for tc in response.tool_calls)
                self.message_history.append(
                    Message(
                        role="assistant",
                        content=f"I'm using these tools: {tool_calls_summary}",
                    )
                )

            # Format and add tool results as user message
            tool_results_content = self._format_tool_results(tool_results)
            self.message_history.append(
                Message(role="user", content=tool_results_content)
            )

        # Check if max turns exceeded
        if self.turn_count >= self.max_turns:
            self.audit_logger.log_event(
                "max_turns_exceeded",
                {"turns": self.turn_count, "tool_calls": self.total_tool_calls},
            )
            self.console.print(
                f"\n[yellow]Warning: Max turns ({self.max_turns}) exceeded[/yellow]"
            )
            if final_response is None:
                final_response = (
                    "I reached the maximum number of turns without completing the task. "
                    "Please try breaking the task into smaller steps."
                )

        # Log session summary
        self.audit_logger.log_event(
            "session_end",
            {
                "total_turns": self.turn_count,
                "total_tool_calls": self.total_tool_calls,
                "session_duration_seconds": time.time() - self.session_start_time,
            },
        )

        return final_response or "Task completed."

    def shutdown(self):
        """Clean up resources and log session end."""
        self.audit_logger.log_event(
            "session_shutdown",
            {"turns": self.turn_count, "tool_calls": self.total_tool_calls},
        )
