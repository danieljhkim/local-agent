"""Sandboxed filesystem connector."""

import os
from pathlib import Path
from typing import List

from ..config.schema import WorkspaceConfig


class FilesystemConnector:
    """Sandboxed filesystem access."""

    def __init__(self, workspace_config: WorkspaceConfig):
        """Initialize filesystem connector.

        Args:
            workspace_config: Workspace configuration with allowed/denied paths
        """
        self.workspace_config = workspace_config
        self.allowed_roots = [
            Path(root).expanduser().resolve()
            for root in workspace_config.allowed_roots
        ]
        self.denied_paths = [
            Path(path).expanduser().resolve()
            for path in workspace_config.denied_paths
        ]

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is allowed for access.

        Args:
            path: Path to check

        Returns:
            True if allowed, False otherwise
        """
        path = Path(path).expanduser().resolve()

        # Check denied paths first
        for denied in self.denied_paths:
            try:
                path.relative_to(denied)
                return False  # Path is under denied directory
            except ValueError:
                continue

        # Check if path is under any allowed root
        for allowed_root in self.allowed_roots:
            try:
                path.relative_to(allowed_root)
                return True  # Path is under allowed root
            except ValueError:
                continue

        return False  # Not under any allowed root

    def validate_path(self, path: str | Path) -> Path:
        """Validate and resolve a path.

        Args:
            path: Path to validate

        Returns:
            Resolved path

        Raises:
            PermissionError: If path is not allowed
        """
        path = Path(path).expanduser().resolve()

        if not self._is_path_allowed(path):
            raise PermissionError(
                f"Access denied: {path} is not within allowed workspace roots"
            )

        return path

    def read_file(self, path: str | Path, start_line: int = 0, end_line: int | None = None) -> str:
        """Read a file with line range support.

        Args:
            path: File path
            start_line: Starting line (0-indexed)
            end_line: Ending line (exclusive), None for end of file

        Returns:
            File contents

        Raises:
            PermissionError: If path is not allowed
            FileNotFoundError: If file doesn't exist
        """
        validated_path = self.validate_path(path)

        if not validated_path.exists():
            raise FileNotFoundError(f"File not found: {validated_path}")

        if not validated_path.is_file():
            raise ValueError(f"Not a file: {validated_path}")

        with open(validated_path, "r") as f:
            lines = f.readlines()

        if end_line is None:
            end_line = len(lines)

        selected_lines = lines[start_line:end_line]
        return "".join(selected_lines)

    def write_file(self, path: str | Path, content: str) -> None:
        """Write content to a file.

        Args:
            path: File path
            content: Content to write

        Raises:
            PermissionError: If path is not allowed
        """
        validated_path = self.validate_path(path)

        # Create parent directories if they don't exist
        validated_path.parent.mkdir(parents=True, exist_ok=True)

        with open(validated_path, "w") as f:
            f.write(content)

    def list_dir(self, path: str | Path) -> List[str]:
        """List directory contents.

        Args:
            path: Directory path

        Returns:
            List of file/directory names

        Raises:
            PermissionError: If path is not allowed
            NotADirectoryError: If path is not a directory
        """
        validated_path = self.validate_path(path)

        if not validated_path.exists():
            raise FileNotFoundError(f"Directory not found: {validated_path}")

        if not validated_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {validated_path}")

        return [item.name for item in validated_path.iterdir()]

    def search_files(
        self, root: str | Path, pattern: str, max_results: int = 100
    ) -> List[str]:
        """Search for files matching a pattern.

        Args:
            root: Root directory to search
            pattern: Glob pattern
            max_results: Maximum number of results

        Returns:
            List of matching file paths

        Raises:
            PermissionError: If path is not allowed
        """
        validated_root = self.validate_path(root)

        if not validated_root.exists():
            raise FileNotFoundError(f"Directory not found: {validated_root}")

        results = []
        for path in validated_root.rglob(pattern):
            if len(results) >= max_results:
                break
            if path.is_file() and self._is_path_allowed(path):
                results.append(str(path))

        return results

    def delete_file(self, path: str | Path) -> None:
        """Delete a file.

        Args:
            path: File path

        Raises:
            PermissionError: If path is not allowed
            FileNotFoundError: If file doesn't exist
        """
        validated_path = self.validate_path(path)

        if not validated_path.exists():
            raise FileNotFoundError(f"File not found: {validated_path}")

        if validated_path.is_file():
            validated_path.unlink()
        else:
            raise ValueError(f"Not a file: {validated_path}")
