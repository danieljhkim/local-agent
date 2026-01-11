"""Policy evaluation engine for tool execution."""

import fnmatch
from typing import Any, Dict

from ..config.schema import AgentConfig, ApprovalPolicy
from ..tools.schema import RiskTier, ToolSchema


class PolicyDecision:
    """Result of a policy evaluation."""

    def __init__(
        self, allowed: bool, requires_approval: bool = False, reason: str | None = None
    ):
        self.allowed = allowed
        self.requires_approval = requires_approval
        self.reason = reason


class PolicyEngine:
    """Engine for evaluating tool execution policies."""

    def __init__(self, config: AgentConfig):
        """Initialize policy engine.

        Args:
            config: Agent configuration
        """
        self.config = config

    def evaluate(
        self, tool: ToolSchema, parameters: Dict[str, Any]
    ) -> PolicyDecision:
        """Evaluate whether a tool call is allowed.

        Args:
            tool: Tool schema
            parameters: Tool parameters

        Returns:
            PolicyDecision indicating if allowed and if approval is required
        """
        # Tier 2 tools always require approval unless policy says otherwise
        if tool.risk_tier == RiskTier.TIER_2:
            # Check for auto-approval policies
            for policy in self.config.approval_policies:
                if self._matches_policy(tool.name, policy):
                    if self._check_conditions(parameters, policy.conditions):
                        if policy.auto_approve:
                            return PolicyDecision(
                                allowed=True,
                                requires_approval=False,
                                reason=f"Auto-approved by policy: {policy.tool_pattern}",
                            )

            # No auto-approval policy, require approval
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Tier 2 tool requires approval",
            )

        # Tier 0 and Tier 1 tools are allowed without approval
        return PolicyDecision(allowed=True, requires_approval=False)

    def _matches_policy(self, tool_name: str, policy: ApprovalPolicy) -> bool:
        """Check if a tool name matches a policy pattern.

        Args:
            tool_name: Tool name
            policy: Approval policy

        Returns:
            True if matches
        """
        return fnmatch.fnmatch(tool_name, policy.tool_pattern)

    def _check_conditions(
        self, parameters: Dict[str, Any], conditions: Dict[str, Any]
    ) -> bool:
        """Check if parameters meet policy conditions.

        Args:
            parameters: Tool parameters
            conditions: Policy conditions

        Returns:
            True if conditions are met
        """
        if not conditions:
            return True

        for key, expected_value in conditions.items():
            if key not in parameters:
                return False

            actual_value = parameters[key]

            # Handle different comparison types
            if isinstance(expected_value, dict):
                # Support operators like {"startswith": "/path"}
                if "startswith" in expected_value:
                    if not str(actual_value).startswith(expected_value["startswith"]):
                        return False
                elif "in" in expected_value:
                    if actual_value not in expected_value["in"]:
                        return False
            else:
                # Direct equality
                if actual_value != expected_value:
                    return False

        return True
