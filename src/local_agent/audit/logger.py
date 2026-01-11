"""Audit logger for tool calls and agent actions."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..config.schema import AuditConfig


class AuditLogger:
    """Logger for auditing tool calls and agent actions."""

    def __init__(self, config: AuditConfig, session_id: str):
        """Initialize audit logger.

        Args:
            config: Audit configuration
            session_id: Unique session identifier
        """
        self.config = config
        self.session_id = session_id
        self.log_dir = Path(config.log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create session log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{timestamp}_{session_id}.jsonl"

    def _redact_sensitive(self, data: Any) -> Any:
        """Redact sensitive information from data.

        Args:
            data: Data to redact

        Returns:
            Redacted data
        """
        if not self.config.enabled:
            return data

        if isinstance(data, dict):
            return {k: self._redact_sensitive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._redact_sensitive(item) for item in data]
        elif isinstance(data, str):
            # Apply redaction patterns
            redacted = data
            for pattern in self.config.redact_patterns:
                redacted = re.sub(pattern, "***REDACTED***", redacted, flags=re.IGNORECASE)
            return redacted
        else:
            return data

    def log_tool_call(
        self,
        tool_name: str,
        risk_tier: str,
        parameters: Dict[str, Any],
        success: bool,
        result: Any = None,
        error: str | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            risk_tier: Risk tier of the tool
            parameters: Tool parameters
            success: Whether the call succeeded
            result: Tool result (will be redacted)
            error: Error message if failed
            elapsed_ms: Execution time in milliseconds
        """
        if not self.config.enabled:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "tool_call",
            "tool_name": tool_name,
            "risk_tier": risk_tier,
            "parameters": self._redact_sensitive(parameters),
            "success": success,
            "result_metadata": self._extract_metadata(result),
            "error": error,
            "elapsed_ms": elapsed_ms,
        }

        self._write_log(log_entry)

    def log_approval(
        self, tool_name: str, approved: bool, reason: str | None = None
    ) -> None:
        """Log an approval decision.

        Args:
            tool_name: Name of the tool
            approved: Whether the action was approved
            reason: Reason for decision
        """
        if not self.config.enabled:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "approval",
            "tool_name": tool_name,
            "approved": approved,
            "reason": reason,
        }

        self._write_log(log_entry)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a generic event.

        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.config.enabled:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            **self._redact_sensitive(data),
        }

        self._write_log(log_entry)

    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract metadata from result without sensitive content.

        Args:
            result: Tool result

        Returns:
            Metadata dictionary
        """
        if result is None:
            return {}

        metadata = {"type": type(result).__name__}

        if isinstance(result, (list, tuple)):
            metadata["count"] = len(result)
        elif isinstance(result, dict):
            metadata["keys"] = list(result.keys())
        elif isinstance(result, str):
            metadata["length"] = len(result)

        return metadata

    def _write_log(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to file.

        Args:
            log_entry: Log entry to write
        """
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
