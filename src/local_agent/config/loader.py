"""Configuration loading utilities."""

from pathlib import Path

import yaml

from .schema import AgentConfig


def load_config(config_path: str | Path | None = None) -> AgentConfig:
    """Load agent configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default locations.

    Returns:
        AgentConfig instance
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path.home() / ".config" / "agent" / "config.yaml",
            Path.home() / ".local" / "agent" / "config.yaml",
            Path.cwd() / ".agent" / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None:
        # No config file found, use defaults
        return AgentConfig()

    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    return AgentConfig(**config_data)


def save_config(config: AgentConfig, config_path: str | Path) -> None:
    """Save agent configuration to YAML file.

    Args:
        config: AgentConfig instance
        config_path: Path to save config file
    """
    config_path = Path(config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump(mode="json")

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
