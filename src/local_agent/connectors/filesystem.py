"""Sandboxed filesystem connector."""

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Any

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

    def apply_patch(self, path: str | Path, unified_diff: str) -> Dict[str, Any]:
        """Apply a unified diff patch to a file.

        This is a Tier 1 operation - it creates a proposed change without actually
        modifying the file. The patch is validated and a preview is returned.

        Args:
            path: File path
            unified_diff: Unified diff format patch

        Returns:
            Dict with:
                - success: bool
                - path: str (resolved path)
                - lines_changed: int
                - hunks_applied: int
                - preview: str (preview of changes)
                - backup_path: str (path to backup file created)

        Raises:
            PermissionError: If path is not allowed
            FileNotFoundError: If file doesn't exist
            ValueError: If patch format is invalid or doesn't apply cleanly
        """
        validated_path = self.validate_path(path)

        if not validated_path.exists():
            raise FileNotFoundError(f"File not found: {validated_path}")

        if not validated_path.is_file():
            raise ValueError(f"Not a file: {validated_path}")

        # Read original file
        with open(validated_path, "r") as f:
            original_lines = f.readlines()

        # Parse and apply patch
        try:
            patched_lines, stats = self._parse_and_apply_unified_diff(
                original_lines, unified_diff
            )
        except Exception as e:
            raise ValueError(f"Failed to apply patch: {str(e)}") from e

        # Create backup of original file
        backup_fd, backup_path = tempfile.mkstemp(
            prefix=f"{validated_path.name}_backup_",
            suffix=".txt",
            dir=validated_path.parent,
        )
        try:
            with os.fdopen(backup_fd, "w") as f:
                f.writelines(original_lines)
        except:
            os.close(backup_fd)
            raise

        # Write patched content to original file
        with open(validated_path, "w") as f:
            f.writelines(patched_lines)

        # Generate preview (first 500 chars of diff)
        preview_lines = []
        for i, (orig, new) in enumerate(zip(original_lines, patched_lines)):
            if orig != new:
                preview_lines.append(f"Line {i+1}:")
                preview_lines.append(f"  - {orig.rstrip()}")
                preview_lines.append(f"  + {new.rstrip()}")
                if len("\n".join(preview_lines)) > 500:
                    preview_lines.append("  ... (truncated)")
                    break

        return {
            "success": True,
            "path": str(validated_path),
            "lines_changed": stats["lines_changed"],
            "hunks_applied": stats["hunks_applied"],
            "preview": "\n".join(preview_lines) if preview_lines else "No changes",
            "backup_path": backup_path,
        }

    def _parse_and_apply_unified_diff(
        self, original_lines: List[str], unified_diff: str
    ) -> tuple[List[str], Dict[str, int]]:
        """Parse unified diff and apply to original lines.

        Args:
            original_lines: Original file lines
            unified_diff: Unified diff string

        Returns:
            Tuple of (patched_lines, stats_dict)

        Raises:
            ValueError: If patch format is invalid or doesn't apply
        """
        # Parse hunks from unified diff
        hunks = self._parse_unified_diff(unified_diff)

        if not hunks:
            raise ValueError("No valid hunks found in patch")

        # Normalize original lines (ensure they all end with \n)
        normalized_lines = [
            line if line.endswith("\n") else line + "\n"
            for line in original_lines
        ]

        # Apply hunks
        result_lines = []
        lines_changed = 0
        hunks_applied = 0
        current_pos = 0  # Current position in original file (0-indexed)

        for hunk in hunks:
            old_start = hunk["old_start"] - 1  # Convert to 0-indexed
            operations = hunk["operations"]

            # Copy lines before this hunk
            result_lines.extend(normalized_lines[current_pos:old_start])
            current_pos = old_start

            # Apply hunk operations
            for op_type, line_content in operations:
                if op_type == " ":
                    # Context line - verify and copy
                    if current_pos >= len(normalized_lines):
                        raise ValueError(f"Context mismatch at line {current_pos + 1}: unexpected end of file")

                    expected = line_content.rstrip()
                    actual = normalized_lines[current_pos].rstrip()

                    if expected != actual:
                        raise ValueError(
                            f"Context mismatch at line {current_pos + 1}: "
                            f"expected '{expected}', got '{actual}'"
                        )
                    result_lines.append(normalized_lines[current_pos])
                    current_pos += 1
                elif op_type == "-":
                    # Line to remove - verify it matches and skip it
                    if current_pos >= len(normalized_lines):
                        raise ValueError(f"Cannot remove line {current_pos + 1}: unexpected end of file")

                    expected = line_content.rstrip()
                    actual = normalized_lines[current_pos].rstrip()

                    if expected != actual:
                        raise ValueError(
                            f"Cannot remove line {current_pos + 1}: "
                            f"expected '{expected}', got '{actual}'"
                        )
                    current_pos += 1
                    lines_changed += 1
                elif op_type == "+":
                    # Line to add
                    result_lines.append(line_content)
                    lines_changed += 1

            hunks_applied += 1

        # Copy remaining lines after all hunks
        result_lines.extend(normalized_lines[current_pos:])

        stats = {
            "lines_changed": lines_changed,
            "hunks_applied": hunks_applied,
        }

        return result_lines, stats

    def _parse_unified_diff(self, unified_diff: str) -> List[Dict[str, Any]]:
        """Parse unified diff format into structured hunks.

        Args:
            unified_diff: Unified diff string

        Returns:
            List of hunk dictionaries with format:
            {
                "old_start": int,  # Starting line in old file (1-indexed)
                "old_count": int,  # Number of lines in old file
                "new_start": int,  # Starting line in new file (1-indexed)
                "new_count": int,  # Number of lines in new file
                "operations": List[Tuple[str, str]]  # (operation_type, line_content)
            }

        Raises:
            ValueError: If format is invalid
        """
        hunks = []
        current_hunk = None
        lines = unified_diff.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip header lines (---, +++, diff, index, etc.)
            if line.startswith("---") or line.startswith("+++") or line.startswith("diff") or line.startswith("index"):
                i += 1
                continue

            # Hunk header: @@ -start,count +start,count @@
            if line.startswith("@@"):
                match = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
                if match:
                    if current_hunk:
                        hunks.append(current_hunk)

                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1

                    current_hunk = {
                        "old_start": old_start,
                        "old_count": old_count,
                        "new_start": new_start,
                        "new_count": new_count,
                        "operations": [],
                    }
                i += 1
                continue

            # Process hunk content if we're in a hunk
            if current_hunk is not None:
                if line.startswith(" "):
                    # Context line (unchanged)
                    current_hunk["operations"].append((" ", line[1:] + "\n"))
                    i += 1
                    continue
                elif line.startswith("-"):
                    # Removal line
                    current_hunk["operations"].append(("-", line[1:] + "\n"))
                    i += 1
                    continue
                elif line.startswith("+"):
                    # Addition line
                    current_hunk["operations"].append(("+", line[1:] + "\n"))
                    i += 1
                    continue

            i += 1

        # Add last hunk
        if current_hunk:
            hunks.append(current_hunk)

        return hunks
