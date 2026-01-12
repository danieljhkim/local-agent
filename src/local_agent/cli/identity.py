"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


identity_app = typer.Typer(help="Manage agent identities (system prompts)")
# app.add_typer(identity_app, name="identity")


@identity_app.command("list")
def identity_list() -> None:
    """List all available identities."""
    from ..identities import get_identity_manager

    manager = get_identity_manager()
    identities = manager.list_identities()
    active = manager.get_active()

    if not identities:
        console.print("[yellow]No identities found[/yellow]")
        return

    console.print("\n[bold cyan]Available Identities[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Status")

    for name in identities:
        is_builtin = manager.is_builtin(name)
        is_active = name == active

        type_str = "[dim]built-in[/dim]" if is_builtin else "custom"
        status = "[green]● active[/green]" if is_active else ""

        table.add_row(name, type_str, status)

    console.print(table)
    console.print(f"\n[dim]Identities location: {manager.identities_dir}[/dim]")


@identity_app.command("current")
def identity_current() -> None:
    """Show the currently active identity."""
    from ..identities import get_identity_manager

    manager = get_identity_manager()
    active = manager.get_active()

    console.print(f"\n[bold cyan]Active Identity:[/bold cyan] {active}")

    if manager.is_builtin(active):
        console.print("[dim]Type: built-in[/dim]")
    else:
        console.print("[dim]Type: custom[/dim]")

    console.print(f"[dim]Path: {manager.get_path(active)}[/dim]")


@identity_app.command("show")
def identity_show(
    name: str = typer.Argument(..., help="Identity name to display"),
) -> None:
    """Display the content of an identity."""
    from ..identities import get_identity_manager

    manager = get_identity_manager()

    try:
        content = manager.get_content(name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Header
    console.print(f"\n[bold cyan]Identity: {name}[/bold cyan]")
    if manager.is_builtin(name):
        console.print("[dim]Type: built-in[/dim]")
    else:
        console.print("[dim]Type: custom[/dim]")
    console.print("[dim]" + "─" * 80 + "[/dim]\n")

    # Content
    console.print(content)

    # Footer
    lines = content.count("\n") + 1
    chars = len(content)
    console.print(f"\n[dim]" + "─" * 80 + "[/dim]")
    console.print(f"[dim]{lines} lines, {chars} characters[/dim]")


@identity_app.command("set")
def identity_set(
    name: str = typer.Argument(..., help="Identity name to set as active"),
) -> None:
    """Set the active identity."""
    from ..identities import get_identity_manager

    manager = get_identity_manager()

    try:
        manager.set_active(name)
        console.print(f"[green]✓[/green] Active identity set to: [cyan]{name}[/cyan]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@identity_app.command("new")
def identity_new(
    name: str = typer.Argument(..., help="Name for the new identity"),
    from_file: Optional[Path] = typer.Option(
        None,
        "--from-file",
        "-f",
        help="Import content from a file instead of interactive input",
    ),
) -> None:
    """Create a new identity with interactive Rich prompt."""
    from rich.prompt import Prompt

    from ..identities import get_identity_manager

    manager = get_identity_manager()

    # Check if already exists
    if manager.exists(name):
        console.print(f"[red]Error:[/red] Identity '{name}' already exists")
        raise typer.Exit(1)

    if from_file:
        # Import from file
        if not from_file.exists():
            console.print(f"[red]Error:[/red] File not found: {from_file}")
            raise typer.Exit(1)

        content = from_file.read_text()
    else:
        # Interactive prompt
        console.print(f"\n[bold cyan]Creating new identity: {name}[/bold cyan]")
        console.print("[dim]Enter your identity/system prompt below.[/dim]")
        console.print(
            "[dim]When finished, enter a blank line followed by 'END' on its own line.[/dim]\n"
        )

        lines = []
        while True:
            try:
                line = console.input("[dim]>[/dim] ")
                if line.strip().upper() == "END" and (not lines or lines[-1] == ""):
                    # Remove trailing empty line if present
                    if lines and lines[-1] == "":
                        lines.pop()
                    break
                lines.append(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        content = "\n".join(lines)

    if not content.strip():
        console.print("[red]Error:[/red] Identity content cannot be empty")
        raise typer.Exit(1)

    try:
        path = manager.create(name, content)
        console.print(f"\n[green]✓[/green] Created identity: [cyan]{name}[/cyan]")
        console.print(f"[dim]Location: {path}[/dim]")

        # Ask if user wants to set as active
        set_active = Prompt.ask(
            "\nSet as active identity?", choices=["y", "n"], default="n"
        )
        if set_active.lower() == "y":
            manager.set_active(name)
            console.print(
                f"[green]✓[/green] Active identity set to: [cyan]{name}[/cyan]"
            )

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@identity_app.command("import")
def identity_import(
    source: Path = typer.Argument(..., help="Path to file to import"),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Name for imported identity (defaults to filename)"
    ),
) -> None:
    """Import an identity from an external file."""
    from ..identities import get_identity_manager

    manager = get_identity_manager()

    if not source.exists():
        console.print(f"[red]Error:[/red] File not found: {source}")
        raise typer.Exit(1)

    # Default name from filename (without extension)
    if name is None:
        name = source.stem

    try:
        path = manager.import_identity(name, source)
        console.print(f"[green]✓[/green] Imported identity: [cyan]{name}[/cyan]")
        console.print(f"[dim]Location: {path}[/dim]")
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@identity_app.command("delete")
def identity_delete(
    name: str = typer.Argument(..., help="Identity name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
) -> None:
    """Delete a custom identity (built-in identities are protected)."""
    from rich.prompt import Confirm

    from ..identities import get_identity_manager

    manager = get_identity_manager()

    if not manager.exists(name):
        console.print(f"[red]Error:[/red] Identity '{name}' not found")
        raise typer.Exit(1)

    if manager.is_builtin(name):
        console.print(f"[red]Error:[/red] Cannot delete built-in identity '{name}'")
        raise typer.Exit(1)

    if not force:
        if not Confirm.ask(f"Delete identity '{name}'?", default=False):
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    try:
        manager.delete(name)
        console.print(f"[green]✓[/green] Deleted identity: [cyan]{name}[/cyan]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
