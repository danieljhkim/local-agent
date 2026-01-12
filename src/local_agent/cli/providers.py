"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


providers_app = typer.Typer(help="Manage LLM providers")
# app.add_typer(providers_app, name="providers")


@providers_app.command("list")
def providers_list(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List available LLM providers."""
    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    if not cfg.providers:
        console.print("[yellow]No providers configured[/yellow]")
        console.print(
            "\n[dim]Add providers to your config file or use 'agent config --init'[/dim]"
        )
        return

    console.print("[bold cyan]Available Providers[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Default", style="green")
    table.add_column("Base URL", style="dim")

    for provider in cfg.providers:
        is_default = "✓" if provider.name == cfg.default_provider else ""
        base_url = provider.base_url or "-"
        table.add_row(provider.name, provider.model, is_default, base_url)

    console.print(table)
    console.print(
        f"\n[dim]Default provider: [cyan]{cfg.default_provider or 'None'}[/cyan][/dim]"
    )


@providers_app.command("set")
def providers_set(
    provider_name: str = typer.Argument(..., help="Provider name to set as default"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Set the default LLM provider."""
    # Determine config file path
    if config_file is None:
        default_paths = [
            Path.home() / ".config" / "local-agent" / "config.yaml",
            Path.home() / ".local" / "share" / "local-agent" / "config.yaml",
            Path.cwd() / ".agent" / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_file = path
                break

    if config_file is None:
        console.print("[red]Error:[/red] No config file found")
        console.print("[dim]Use 'agent config --init' to create one[/dim]")
        raise typer.Exit(1)

    # Load config
    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Check if provider exists
    provider_names = [p.name for p in cfg.providers]
    if provider_name not in provider_names:
        console.print(f"[red]Error:[/red] Provider '{provider_name}' not found")
        console.print(f"\n[yellow]Available providers:[/yellow]")
        for name in provider_names:
            console.print(f"  - {name}")
        raise typer.Exit(1)

    # Update default provider
    cfg.default_provider = provider_name

    # Save config
    try:
        save_config(cfg, config_file)
        console.print(
            f"[green]✓[/green] Default provider set to: [cyan]{provider_name}[/cyan]"
        )
        console.print(f"[dim]Config saved to: {config_file}[/dim]")
    except Exception as e:
        console.print(f"[red]Error saving config:[/red] {e}")
        raise typer.Exit(1)


# Logs command group
