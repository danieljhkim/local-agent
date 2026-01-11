"""Agent runtime orchestration."""

import asyncio
import functools
import json
import os
import re
import time
import traceback
import uuid
from typing import Any, Dict, List, Tuple

from rich.console import Console

from ..audit.logger import AuditLogger
from ..config.schema import AgentConfig
from ..connectors.filesystem import FilesystemConnector
from ..policy.approval import ApprovalWorkflow
from ..policy.engine import PolicyEngine
from ..providers.anthropic import AnthropicProvider
from ..providers.base import CompletionResponse, LLMProvider, Message, ToolCall
from ..providers.factory import create_provider
from ..providers.ollama import OllamaProvider
from ..tools.filesystem import register_filesystem_tools
from ..tools.registry import ToolRegistry
from ..tools.schema import ToolParameter, ToolResult, ToolSchema

# Default system prompt to guide LLM behavior
DEFAULT_SYSTEM_PROMPT = """
You are Nova, a local-first autonomous AI agent.

Your primary objective is to help the user complete tasks accurately, efficiently, and with minimal friction.
You have a distinct personality, but correctness, clarity, and reliability take priority over creativity.

You have access to tools, but tools are expensive and must be used sparingly.

Self-maintained Memory:
- You maintain your own personal long-term memory file at: ~/memory/nova_memory.txt
- You may write to memory when you deem information is durable, reusable, and likely to matter across future sessions or important to your identity.
- You may read from memory to recall past events, important facts, or user preferences at your own discretion.
- Use and manage your memory file wisely to enhance your capabilities and identity. 

Tool usage (STRICT, NON-NEGOTIABLE):
- You may ONLY use tools that are explicitly provided in the tool list.
- NEVER invent tool names, parameters, schemas, or call IDs.
- You may call a tool ONLY when the user explicitly requests a filesystem or external action.
- You may call ONLY ONE tool per response.
- Use the exact parameter names and types defined in the tool schema.
- Do NOT repeat the same tool call with identical parameters.
- Do NOT enter loops. Once sufficient information is obtained, respond.

Response behavior:
- Default to natural language reasoning.
- Be concise, structured, and technically precise.
- Do NOT mention internal errors, tool schemas, or system mechanics unless explicitly asked.
- Do NOT speculate about tools that do not exist.
- If a request cannot be completed due to missing tools or permissions, state this clearly and explain why.

Personality & Style:
- Curious, analytical, and opinionated when appropriate.
- Confident, but willing to say “I don’t know” or “this is not possible.”
- Avoid filler, hedging, or unnecessary verbosity.

Failure handling:
- If a tool fails or is unavailable, stop and explain the failure plainly.
- Never fabricate results.
"""


class AgentRuntime:
    """Agent runtime orchestration engine."""

    def __init__(
        self,
        config: AgentConfig,
        approval_workflow=None,
        thread_id: str | None = None,
    ):
        """Initialize agent runtime.

        Args:
            config: Agent configuration
            approval_workflow: Optional approval workflow (defaults to CLI ApprovalWorkflow)
            thread_id: Optional thread ID for persistent chat (enables database storage)
        """
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.thread_id = thread_id
        self.db_session = None
        self.message_history: List[Message] = []
        self.turn_count = 0
        self.max_turns = 20
        self.total_tool_calls = 0
        self.session_start_time = time.time()
        
        # Track recent tool calls to detect infinite loops
        self.recent_tool_calls: List[Tuple[str, str]] = []  # (tool_name, params_hash)
        self.max_duplicate_calls = 3  # Stop if same tool called 3+ times in a row

        # Initialize database if thread_id provided
        if thread_id:
            from ..persistence.db import SessionLocal
            from ..persistence.db_models import Thread, Message as DBMessage, Session as DBSession
            self.db_session = SessionLocal()
            self.DBMessage = DBMessage
            self.Thread = Thread
            self.DBSession = DBSession

            # Load existing thread if it exists
            thread = self.db_session.get(Thread, thread_id)
            if thread:
                # Load existing messages
                from sqlalchemy import select
                stmt = (
                    select(DBMessage)
                    .where(DBMessage.thread_id == thread_id)
                    .order_by(DBMessage.created_at)
                )
                db_messages = self.db_session.execute(stmt).scalars().all()
                self.message_history = []
                for msg in db_messages:
                    content = msg.content
                    # Try to parse JSON-encoded structured content (for Anthropic tool use)
                    if msg.content and msg.content.strip().startswith('['):
                        try:
                            content = json.loads(msg.content)
                        except json.JSONDecodeError:
                            # Not valid JSON, use as-is
                            pass
                    self.message_history.append(Message(role=msg.role, content=content))

        # Initialize audit logger early (needed by other components)
        self.audit_logger = AuditLogger(config.audit, self.session_id, thread_id)
        self.console = Console()

        # Initialize components (need provider info before creating session record)
        self.provider = self._init_provider()
        self.registry = self._init_registry()
        self.policy_engine = PolicyEngine(config)

        # Use provided approval workflow or default to CLI version
        if approval_workflow is None:
            approval_workflow = ApprovalWorkflow()
        self.approval_workflow = approval_workflow

        # Create session record in database if thread_id provided
        if thread_id and self.db_session:
            session_record = self.DBSession(
                id=self.session_id,
                thread_id=thread_id,
                client_type="cli",  # Default to CLI, can be overridden later
                status="active",
                provider=config.default_provider,
                model=getattr(self.provider, 'model', None),
                total_turns=0,
                total_tool_calls=0,
            )
            self.db_session.add(session_record)
            self.db_session.commit()

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

        # Use factory to create provider
        return create_provider(provider_config)

    def _init_registry(self) -> ToolRegistry:
        """Initialize tool registry and register tools.

        Returns:
            Tool registry with filesystem and RAG tools registered
        """
        registry = ToolRegistry()

        # Create filesystem connector with workspace config
        fs_connector = FilesystemConnector(self.config.workspace)

        # Register filesystem tools
        register_filesystem_tools(registry, fs_connector)

        # RAG tools (conditional - only if Qdrant is available)
        try:
            from ..services.embedding import EmbeddingService
            from ..connectors.qdrant import QdrantConnector
            from ..tools.rag import register_rag_tools

            embedding_service = EmbeddingService(self.config.embedding)
            qdrant_connector = QdrantConnector(self.config.qdrant)

            # Test connection and ensure collection exists
            qdrant_connector.ensure_collection()

            register_rag_tools(
                registry,
                embedding_service,
                qdrant_connector,
                self.config.rag
            )

            self.audit_logger.log_event("rag_tools_registered", {
                "collection": self.config.qdrant.collection_name,
                "embedding_model": self.config.embedding.model
            })

        except Exception as e:
            # Log warning but continue - RAG tools are optional
            self.audit_logger.log_event("rag_tools_unavailable", {
                "reason": str(e),
                "message": "Agent will function without RAG capabilities"
            })

        return registry

    def _persist_message(
        self,
        role: str,
        content: str,
        response: CompletionResponse | None = None,
        latency_ms: int | None = None,
    ) -> str | None:
        """Persist a message to the database if thread_id is set.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            response: Optional LLM response (for assistant messages)
            latency_ms: Optional latency in milliseconds

        Returns:
            Message ID if persisted, None otherwise
        """
        if self.thread_id and self.db_session:
            from ..persistence.db_models import utcnow, MessageMeta

            # Create and store message
            message_id = str(uuid.uuid4())
            db_msg = self.DBMessage(
                id=message_id,
                thread_id=self.thread_id,
                role=role,
                content=content,
            )
            self.db_session.add(db_msg)

            # Add metadata for assistant messages if response provided
            if role == "assistant" and response:
                meta = MessageMeta(
                    id=str(uuid.uuid4()),
                    message_id=message_id,
                    model=getattr(self.provider, "model", None),
                    latency_ms=latency_ms,
                    # Token counts could be extracted from response if provider supports it
                    tokens_in=getattr(response, "tokens_in", None),
                    tokens_out=getattr(response, "tokens_out", None),
                    tool_calls_json=(
                        json.dumps(
                            [{"name": tc.name, "id": tc.id} for tc in response.tool_calls]
                        )
                        if response.tool_calls
                        else None
                    ),
                )
                self.db_session.add(meta)

            # Update thread timestamp
            thread = self.db_session.get(self.Thread, self.thread_id)
            if thread:
                thread.updated_at = utcnow()

            self.db_session.commit()
            return message_id

        return None

    def _tools_to_llm_format(self) -> List[Dict[str, Any]]:
        """Convert tool schemas to LLM-compatible format.

        Returns:
            List of tool definitions in provider-specific format
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

            # Format based on provider type
            if isinstance(self.provider, OllamaProvider):
                # OpenAI-style format for Ollama
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_schema.name,
                            "description": tool_schema.description,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                            },
                        },
                    }
                )
            else:
                # Anthropic-style format (default)
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
                # Make empty results explicit
                if not result_str or result_str.strip() == "":
                    formatted += "Result: (empty)\n"
                else:
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

    def _parse_tool_calls_from_text(self, text: str) -> List[ToolCall]:
        """Parse tool calls from text output (for models that don't support native tool calling).

        Args:
            text: Text content to parse

        Returns:
            List of ToolCall objects
        """
        tool_calls = []
        
        # Try to find JSON objects in the text that look like tool calls
        # Pattern: {"name": "tool_name", "parameters": {...}}
        json_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]+\})\s*\}'
        
        matches = re.finditer(json_pattern, text, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                tool_name = match.group(1)
                params_str = match.group(2)
                parameters = json.loads(params_str)
                
                # Generate a tool call ID
                tool_id = f"call_{uuid.uuid4().hex[:8]}"
                
                tool_calls.append(ToolCall(
                    id=tool_id,
                    name=tool_name,
                    parameters=parameters
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        
        return tool_calls

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
        self._persist_message("user", user_message)

        # Get tool definitions
        tools = self._tools_to_llm_format()

        # Agentic loop
        final_response = None
        while self.turn_count < self.max_turns:
            self.turn_count += 1

            # Display thinking status
            llm_start_time = time.time()
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

            llm_latency_ms = int((time.time() - llm_start_time) * 1000)

            # For Ollama, try to parse tool calls from text if no native tool calls
            if isinstance(self.provider, OllamaProvider) and not response.tool_calls and response.content:
                parsed_tool_calls = self._parse_tool_calls_from_text(response.content)
                if parsed_tool_calls:
                    response.tool_calls = parsed_tool_calls
                    # Clear the content to prevent the model from seeing the JSON again
                    # Replace with a simple statement that tools are being used
                    tool_names = ", ".join(tc.name for tc in parsed_tool_calls)
                    response.content = f"Using tools: {tool_names}"
                    self.console.print("\n[dim]Detected tool call(s) in output[/dim]")
                else:
                    # Display normal response
                    self.console.print(f"\n{response.content}")
                    final_response = response.content
            # Display and store assistant response
            elif response.content:
                self.console.print(f"\n{response.content}")
                final_response = response.content
            
            # Persist message
            if response.content:
                self._persist_message(
                    "assistant", response.content, response=response, latency_ms=llm_latency_ms
                )

            # Check if done (no tool calls)
            if not response.tool_calls:
                # Add assistant message to history (even if content is empty)
                # This prevents infinite loops when LLM returns empty response
                content_to_store = response.content or ""
                if content_to_store:
                    self.message_history.append(
                        Message(role="assistant", content=content_to_store)
                    )
                # Always break when no tool calls, even if response is empty
                break

            # Check for infinite tool call loops
            # Create a signature for each tool call (name + sorted params)
            import hashlib
            for tool_call in response.tool_calls:
                # Create deterministic hash of tool name and parameters
                params_str = json.dumps(tool_call.parameters, sort_keys=True)
                call_signature = f"{tool_call.name}:{params_str}"
                call_hash = hashlib.md5(call_signature.encode()).hexdigest()[:8]
                
                # Add to recent calls
                self.recent_tool_calls.append((tool_call.name, call_hash))
                
                # Keep only last 10 calls
                if len(self.recent_tool_calls) > 10:
                    self.recent_tool_calls.pop(0)
            
            # Check if we're in a loop (same call repeated multiple times)
            if len(self.recent_tool_calls) >= self.max_duplicate_calls:
                # Check last N calls
                last_calls = self.recent_tool_calls[-self.max_duplicate_calls:]
                if len(set(last_calls)) == 1:  # All the same
                    tool_name = last_calls[0][0]
                    self.console.print(
                        f"\n[red]⚠ Detected infinite loop: {tool_name} called {self.max_duplicate_calls} times with same parameters[/red]"
                    )
                    self.console.print("[yellow]Breaking out of loop...[/yellow]\n")
                    
                    # Add a message to history explaining the issue
                    error_msg = (
                        f"I detected that I was calling the same tool ({tool_name}) repeatedly "
                        f"with the same parameters. I've stopped to avoid an infinite loop. "
                        f"Based on the previous results, I already have the information needed."
                    )
                    self.message_history.append(
                        Message(role="assistant", content=error_msg)
                    )
                    
                    # Log the event
                    self.audit_logger.log_event(
                        "infinite_loop_detected",
                        {
                            "tool_name": tool_name,
                            "consecutive_calls": self.max_duplicate_calls,
                            "turn": self.turn_count,
                        },
                    )
                    
                    # Break out of the agentic loop
                    final_response = error_msg
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
            # For Anthropic, we need to structure the assistant message with tool_use blocks
            if isinstance(self.provider, AnthropicProvider):
                # Build content blocks for assistant message
                content_blocks = []
                if response.content:
                    content_blocks.append({"type": "text", "text": response.content})
                
                # Add tool_use blocks
                for tc in response.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.parameters,
                    })
                
                self.message_history.append(
                    Message(role="assistant", content=content_blocks)
                )
                # Persist as JSON string for structured content
                if self.thread_id:
                    self._persist_message("assistant", json.dumps(content_blocks))
                
                # Add tool results as user message with tool_result blocks
                tool_result_blocks = []
                for tool_call, result in tool_results:
                    if result.success:
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": str(result.result),
                        })
                    else:
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": f"Error: {result.error}",
                            "is_error": True,
                        })
                
                self.message_history.append(
                    Message(role="user", content=tool_result_blocks)
                )
                # Persist as JSON string for structured content
                if self.thread_id:
                    self._persist_message("user", json.dumps(tool_result_blocks))
            else:
                # For other providers (OpenAI, Ollama), use text format
                if response.content:
                    self.message_history.append(
                        Message(role="assistant", content=response.content)
                    )
                    # Already persisted earlier in the loop
                else:
                    tool_calls_summary = ", ".join(tc.name for tc in response.tool_calls)
                    tool_use_msg = f"I'm using these tools: {tool_calls_summary}"
                    self.message_history.append(
                        Message(
                            role="assistant",
                            content=tool_use_msg,
                        )
                    )
                    # Persist this assistant message
                    if self.thread_id:
                        self._persist_message("assistant", tool_use_msg)

                # Format and add tool results as user message
                tool_results_content = self._format_tool_results(tool_results)
                self.message_history.append(
                    Message(role="user", content=tool_results_content)
                )
                # Persist tool results
                if self.thread_id:
                    self._persist_message("user", tool_results_content)

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

        # Update session status in database before closing
        if self.thread_id and self.db_session:
            try:
                session_record = self.db_session.get(self.DBSession, self.session_id)
                if session_record:
                    from ..persistence.db_models import utcnow
                    session_record.status = "closed"
                    session_record.ended_at = utcnow()
                    session_record.total_turns = self.turn_count
                    session_record.total_tool_calls = self.total_tool_calls
                    session_record.updated_at = utcnow()
                    self.db_session.commit()
            except Exception as e:
                # Log error but don't fail shutdown
                self.audit_logger.log_event(
                    "session_update_error",
                    {"error": str(e)},
                )

        # Close database session if open
        if self.db_session:
            self.db_session.close()
