"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


memory_app = typer.Typer(help="View and manage Nova's memory")
# app.add_typer(memory_app, name="memory")


@memory_app.command("view")
def memory_view() -> None:
    """Display the contents of Nova's memory file."""
    memory_path = (
        Path.home() / ".local" / "share" / "local-agent" / "memory" / "nova.txt"
    )

    if not memory_path.exists():
        console.print(f"[yellow]Memory file not found:[/yellow] {memory_path}")
        console.print(
            "[dim]Nova's memory file will be created when the agent writes to it[/dim]"
        )
        return

    try:
        content = memory_path.read_text()

        if not content.strip():
            console.print(f"[yellow]Memory file is empty[/yellow]")
            console.print(f"[dim]Location: {memory_path}[/dim]")
            return

        # Display header
        console.print(f"\n[bold cyan]Nova's Memory[/bold cyan]")
        console.print(f"[dim]Location: {memory_path}[/dim]")
        console.print("[dim]" + "─" * 80 + "[/dim]\n")

        # Display content
        console.print(content)

        # Display footer with stats
        lines = content.count("\n") + 1
        chars = len(content)
        console.print(f"\n[dim]" + "─" * 80 + "[/dim]")
        console.print(f"[dim]{lines} lines, {chars} characters[/dim]")

    except Exception as e:
        console.print(f"[red]Error reading memory file:[/red] {e}")
        raise typer.Exit(1)


# Identity command group
