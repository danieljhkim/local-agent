"""Filesystem tools implementation."""

from pathlib import Path

from ..connectors.filesystem import FilesystemConnector
from ..tools.schema import RiskTier, ToolParameter, ToolResult


class FilesystemTools:
    """Filesystem tool implementations."""

    def __init__(self, connector: FilesystemConnector):
        """Initialize filesystem tools.

        Args:
            connector: Filesystem connector instance
        """
        self.connector = connector

    def fs_read_file(
        self,
        path: str,
        start_line: int = 0,
        end_line: int | None = None,
    ) -> ToolResult:
        """Read a file with optional line range.

        Args:
            path: Path to file
            start_line: Starting line (0-indexed)
            end_line: Ending line (exclusive), None for end of file

        Returns:
            ToolResult with file contents
        """
        try:
            content = self.connector.read_file(path, start_line, end_line)
            return ToolResult(
                success=True,
                result=content,
                metadata={
                    "path": str(path),
                    "lines": len(content.splitlines()),
                    "chars": len(content),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def fs_list_dir(self, path: str) -> ToolResult:
        """List directory contents.

        Args:
            path: Directory path

        Returns:
            ToolResult with list of files/directories
        """
        try:
            items = self.connector.list_dir(path)
            return ToolResult(
                success=True,
                result=items,
                metadata={"path": str(path), "count": len(items)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def fs_search(
        self, root: str, pattern: str, max_results: int = 100
    ) -> ToolResult:
        """Search for files matching a pattern.

        Args:
            root: Root directory to search
            pattern: Glob pattern (e.g., "*.py", "**/*.txt")
            max_results: Maximum number of results

        Returns:
            ToolResult with list of matching paths
        """
        try:
            results = self.connector.search_files(root, pattern, max_results)
            return ToolResult(
                success=True,
                result=results,
                metadata={
                    "root": str(root),
                    "pattern": pattern,
                    "count": len(results),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def fs_write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file.

        Args:
            path: File path
            content: Content to write

        Returns:
            ToolResult
        """
        try:
            self.connector.write_file(path, content)
            return ToolResult(
                success=True,
                result=f"Wrote {len(content)} characters to {path}",
                metadata={
                    "path": str(path),
                    "chars": len(content),
                    "lines": len(content.splitlines()),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def fs_delete_file(self, path: str) -> ToolResult:
        """Delete a file.

        Args:
            path: File path

        Returns:
            ToolResult
        """
        try:
            self.connector.delete_file(path)
            return ToolResult(
                success=True,
                result=f"Deleted {path}",
                metadata={"path": str(path)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def fs_apply_patch(self, path: str, patch: str) -> ToolResult:
        """Apply a unified diff patch to a file.

        This is safer than full file rewrites as it only modifies specific lines.
        Creates a backup of the original file before applying changes.

        Args:
            path: File path
            patch: Unified diff format patch

        Returns:
            ToolResult with patch application details
        """
        try:
            result = self.connector.apply_patch(path, patch)
            return ToolResult(
                success=True,
                result=f"Applied patch to {path}. Backup created at {result['backup_path']}",
                metadata={
                    "path": result["path"],
                    "lines_changed": result["lines_changed"],
                    "hunks_applied": result["hunks_applied"],
                    "backup_path": result["backup_path"],
                    "preview": result["preview"],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def register_filesystem_tools(
    registry, connector: FilesystemConnector
) -> None:
    """Register filesystem tools with the registry.

    Args:
        registry: Tool registry
        connector: Filesystem connector
    """
    tools = FilesystemTools(connector)

    # Register read-only tools (Tier 0)
    registry.register(
        name="fs_read_file",
        description="Read contents of a file with optional line range",
        risk_tier=RiskTier.TIER_0,
        handler=tools.fs_read_file,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file",
                required=True,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="Starting line (0-indexed)",
                required=False,
                default=0,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="Ending line (exclusive)",
                required=False,
                default=None,
            ),
        ],
    )

    registry.register(
        name="fs_list_dir",
        description="List contents of a directory",
        risk_tier=RiskTier.TIER_0,
        handler=tools.fs_list_dir,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Directory path",
                required=True,
            ),
        ],
    )

    registry.register(
        name="fs_search",
        description="Search for files matching a glob pattern",
        risk_tier=RiskTier.TIER_0,
        handler=tools.fs_search,
        parameters=[
            ToolParameter(
                name="root",
                type="string",
                description="Root directory to search",
                required=True,
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Glob pattern (e.g., '*.py')",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results",
                required=False,
                default=100,
            ),
        ],
    )

    # Register patch tool (Tier 1 - drafting/proposed changes)
    registry.register(
        name="fs_apply_patch",
        description="Apply a unified diff patch to a file. Safer than full rewrites, creates backup automatically.",
        risk_tier=RiskTier.TIER_1,
        handler=tools.fs_apply_patch,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file",
                required=True,
            ),
            ToolParameter(
                name="patch",
                type="string",
                description="Unified diff format patch",
                required=True,
            ),
        ],
    )

    # Register write tools (Tier 2 - requires approval)
    registry.register(
        name="fs_write_file",
        description="Write content to a file",
        risk_tier=RiskTier.TIER_2,
        handler=tools.fs_write_file,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write",
                required=True,
            ),
        ],
    )

    registry.register(
        name="fs_delete_file",
        description="Delete a file",
        risk_tier=RiskTier.TIER_2,
        handler=tools.fs_delete_file,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file",
                required=True,
            ),
        ],
    )
