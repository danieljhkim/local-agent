"""Identity loader and manager for system prompts."""

import shutil
from pathlib import Path
from typing import List, Optional

# Package-level default identities directory
DEFAULTS_DIR = Path(__file__).parent / "defaults"

# User directories
USER_IDENTITIES_DIR = Path.home() / ".local" / "share" / "local-agent" / "identities"
ACTIVE_IDENTITY_FILE = Path.home() / ".config" / "local-agent" / "identity.active"

# Built-in identity names (protected from deletion)
BUILTIN_IDENTITIES = {"default", "nova"}


class IdentityManager:
    """Manages identity files and active identity selection."""

    def __init__(
        self,
        identities_dir: Path | None = None,
        active_file: Path | None = None,
    ):
        """Initialize identity manager.

        Args:
            identities_dir: Directory for user identities (default: ~/.local/share/local-agent/identities/)
            active_file: File storing active identity name (default: ~/.config/local-agent/identity.active)
        """
        self.identities_dir = identities_dir or USER_IDENTITIES_DIR
        self.active_file = active_file or ACTIVE_IDENTITY_FILE
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure identity directories exist and built-ins are copied."""
        # Create directories
        self.identities_dir.mkdir(parents=True, exist_ok=True)
        self.active_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy built-in identities if they don't exist
        if DEFAULTS_DIR.exists():
            for default_file in DEFAULTS_DIR.glob("*.txt"):
                target = self.identities_dir / default_file.name
                if not target.exists():
                    shutil.copy(default_file, target)

        # Set default active identity if none set
        if not self.active_file.exists():
            self.set_active("default")

    def list_identities(self) -> List[str]:
        """List all available identity names.

        Returns:
            List of identity names (without .txt extension)
        """
        identities = []
        for f in self.identities_dir.glob("*.txt"):
            identities.append(f.stem)
        return sorted(identities)

    def get_active(self) -> str:
        """Get the name of the active identity.

        Returns:
            Active identity name
        """
        if self.active_file.exists():
            name = self.active_file.read_text().strip()
            if name and self.exists(name):
                return name
        return "default"

    def set_active(self, name: str) -> None:
        """Set the active identity.

        Args:
            name: Identity name to set as active

        Raises:
            ValueError: If identity doesn't exist
        """
        if not self.exists(name):
            raise ValueError(f"Identity '{name}' not found")
        self.active_file.write_text(name)

    def exists(self, name: str) -> bool:
        """Check if an identity exists.

        Args:
            name: Identity name

        Returns:
            True if identity exists
        """
        return (self.identities_dir / f"{name}.txt").exists()

    def is_builtin(self, name: str) -> bool:
        """Check if an identity is a built-in (protected).

        Args:
            name: Identity name

        Returns:
            True if identity is built-in
        """
        return name.lower() in BUILTIN_IDENTITIES

    def get_content(self, name: str) -> str:
        """Get the content of an identity file.

        Args:
            name: Identity name

        Returns:
            Identity content (system prompt)

        Raises:
            ValueError: If identity doesn't exist
        """
        path = self.identities_dir / f"{name}.txt"
        if not path.exists():
            raise ValueError(f"Identity '{name}' not found")
        return path.read_text()

    def get_active_content(self) -> str:
        """Get the content of the active identity.

        Returns:
            Active identity content (system prompt)
        """
        return self.get_content(self.get_active())

    def create(self, name: str, content: str) -> Path:
        """Create a new identity.

        Args:
            name: Identity name
            content: Identity content (system prompt)

        Returns:
            Path to created identity file

        Raises:
            ValueError: If identity already exists or name is invalid
        """
        # Validate name
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Identity name must be alphanumeric (underscores and hyphens allowed)"
            )

        if self.exists(name):
            raise ValueError(f"Identity '{name}' already exists")

        path = self.identities_dir / f"{name}.txt"
        path.write_text(content)
        return path

    def update(self, name: str, content: str) -> Path:
        """Update an existing identity.

        Args:
            name: Identity name
            content: New identity content

        Returns:
            Path to updated identity file

        Raises:
            ValueError: If identity doesn't exist
        """
        if not self.exists(name):
            raise ValueError(f"Identity '{name}' not found")

        path = self.identities_dir / f"{name}.txt"
        path.write_text(content)
        return path

    def delete(self, name: str) -> None:
        """Delete an identity.

        Args:
            name: Identity name

        Raises:
            ValueError: If identity doesn't exist or is built-in
        """
        if not self.exists(name):
            raise ValueError(f"Identity '{name}' not found")

        if self.is_builtin(name):
            raise ValueError(f"Cannot delete built-in identity '{name}'")

        path = self.identities_dir / f"{name}.txt"
        path.unlink()

        # If deleted identity was active, switch to default
        if self.get_active() == name:
            self.set_active("default")

    def import_identity(self, name: str, source_path: Path) -> Path:
        """Import an identity from an external file.

        Args:
            name: Name for the imported identity
            source_path: Path to source file

        Returns:
            Path to imported identity file

        Raises:
            ValueError: If name invalid or file not found
            FileNotFoundError: If source file doesn't exist
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        content = source_path.read_text()
        return self.create(name, content)

    def get_path(self, name: str) -> Path:
        """Get the file path for an identity.

        Args:
            name: Identity name

        Returns:
            Path to identity file

        Raises:
            ValueError: If identity doesn't exist
        """
        if not self.exists(name):
            raise ValueError(f"Identity '{name}' not found")
        return self.identities_dir / f"{name}.txt"


# Singleton instance
_manager: Optional[IdentityManager] = None


def get_identity_manager() -> IdentityManager:
    """Get the global identity manager instance.

    Returns:
        IdentityManager singleton
    """
    global _manager
    if _manager is None:
        _manager = IdentityManager()
    return _manager
