"""CLI entry point for the local agent."""

import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config.loader import load_config, save_config
from .config.schema import AgentConfig

app = typer.Typer(
    name="agent",
    help="Local AI agent with tool support and safety-first design",
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Start an interactive chat session with the agent."""
    import asyncio

    from .runtime import AgentRuntime

    console.print("[bold cyan]Local Agent - Interactive Chat[/bold cyan]")
    console.print("[dim]Type 'exit', 'quit', or press Ctrl+D to quit[/dim]\n")

    # Load config
    try:
        config = load_config(config_file)
        console.print(f"[green]✓[/green] Configuration loaded")
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize runtime
    try:
        runtime = AgentRuntime(config)
        console.print(f"[green]✓[/green] Agent initialized\n")
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        raise typer.Exit(1)

    # REPL loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold blue]You:[/bold blue] ")

            # Check for exit commands
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            # Execute task
            console.print()
            response = asyncio.run(runtime.execute(user_input))

            # Display final response if not already shown
            if response:
                console.print(f"\n[bold green]Agent:[/bold green] {response}\n")

        except EOFError:
            # Ctrl+D pressed
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except KeyboardInterrupt:
            # Ctrl+C pressed
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]You can continue or type 'exit' to quit[/dim]")


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description for the agent"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Run a single task and exit."""
    import asyncio

    from .runtime import AgentRuntime

    console.print(f"[bold cyan]Task:[/bold cyan] {task}\n")

    # Load config
    try:
        config = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize runtime
    try:
        runtime = AgentRuntime(config)
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        raise typer.Exit(1)

    # Execute task
    try:
        response = asyncio.run(runtime.execute(task))
        console.print(f"\n[bold green]Result:[/bold green]\n{response}")
    except Exception as e:
        console.print(f"[red]Error executing task:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def tools(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List available tools and their permissions."""
    console.print("[yellow]Available Tools[/yellow]\n")

    # Load config to get workspace settings
    try:
        config = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize registry and register tools
    from .tools.registry import ToolRegistry
    from .tools.filesystem import register_filesystem_tools
    from .connectors.filesystem import FilesystemConnector

    registry = ToolRegistry()
    fs_connector = FilesystemConnector(config.workspace)
    register_filesystem_tools(registry, fs_connector)

    tools_list = registry.list_tools()

    if not tools_list:
        console.print("[dim]No tools registered yet[/dim]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Risk Tier", style="yellow")
    table.add_column("Description")

    for tool in tools_list:
        table.add_row(tool.name, tool.risk_tier.value, tool.description)

    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current config"),
    init: bool = typer.Option(False, "--init", help="Initialize a new config file"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Manage agent configuration."""

    if init:
        # Create a default config file
        default_config = AgentConfig()

        if config_file is None:
            config_file = Path.home() / ".config" / "agent" / "config.yaml"

        if config_file.exists():
            overwrite = typer.confirm(
                f"Config file already exists at {config_file}. Overwrite?"
            )
            if not overwrite:
                raise typer.Exit(0)

        save_config(default_config, config_file)
        console.print(f"[green]✓[/green] Created config file at: {config_file}")
        console.print("\n[yellow]Next steps:[/yellow]")
        console.print("1. Edit the config file to add your LLM provider API keys")
        console.print("2. Configure allowed workspace roots")
        console.print("3. Set up approval policies")
        return

    if show:
        try:
            cfg = load_config(config_file)
            console.print("[yellow]Current Configuration:[/yellow]\n")
            console.print(f"[cyan]Providers:[/cyan] {len(cfg.providers)}")
            for provider in cfg.providers:
                console.print(f"  - {provider.name}: {provider.model}")
            console.print(f"\n[cyan]Workspace Roots:[/cyan]")
            for root in cfg.workspace.allowed_roots:
                console.print(f"  - {root}")
            console.print(f"\n[cyan]Approval Policies:[/cyan] {len(cfg.approval_policies)}")
            console.print(f"[cyan]Audit Enabled:[/cyan] {cfg.audit.enabled}")
            console.print(f"[cyan]State Dir:[/cyan] {cfg.state_dir}")
        except Exception as e:
            console.print(f"[red]Error loading config:[/red] {e}")
            raise typer.Exit(1)
        return

    # Default: show help
    console.print("[yellow]Config Management[/yellow]\n")
    console.print("Use --init to create a new config file")
    console.print("Use --show to view current configuration")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[cyan]local-agent[/cyan] version [green]{__version__}[/green]")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
