"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


threads_app = typer.Typer(help="Manage conversation threads")
# app.add_typer(threads_app, name="threads")


@threads_app.command("list")
def threads_list(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of threads to show"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List recent conversation threads."""
    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Thread
    from ..persistence.database_init import check_database_exists, init_database
    from sqlalchemy import select

    # Initialize database if it doesn't exist
    if not check_database_exists():
        console.print("[yellow]Database not found. Initializing...[/yellow]")
        init_database()

    db = SessionLocal()
    try:
        stmt = select(Thread).order_by(Thread.updated_at.desc()).limit(limit)
        threads = db.execute(stmt).scalars().all()

        if not threads:
            console.print("[yellow]No threads found[/yellow]")
            console.print("\n[dim]Create a new thread with 'agent threads new'[/dim]")
            return

        console.print(
            f"[bold cyan]Recent Threads[/bold cyan] (showing {len(threads)} of {limit})\n"
        )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Title", style="yellow")
        table.add_column("Messages", style="dim", width=10)
        table.add_column("Created", style="dim", width=16)
        table.add_column("Updated", style="dim", width=16)

        for thread in threads:
            message_count = len(thread.messages)
            table.add_row(
                thread.id[:8] + "...",
                thread.title[:50] + ("..." if len(thread.title) > 50 else ""),
                str(message_count),
                thread.created_at.strftime("%Y-%m-%d %H:%M"),
                thread.updated_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print(
            f"\n[dim]Use 'agent threads resume <id>' to continue a conversation[/dim]"
        )
    finally:
        db.close()


@threads_app.command("new")
def threads_new(
    title: str = typer.Option("New conversation", "--title", "-t", help="Thread title"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Create a new conversation thread and start chatting."""
    import asyncio
    import uuid

    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Thread
    from ..persistence.database_init import check_database_exists, init_database
    from ..runtime import AgentRuntime

    # Initialize database if it doesn't exist
    if not check_database_exists():
        console.print("[yellow]Database not found. Initializing...[/yellow]")
        init_database()

    # Create new thread in database
    thread_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        thread = Thread(id=thread_id, title=title)
        db.add(thread)
        db.commit()
        console.print(f"[green]✓[/green] Created new thread: [cyan]{title}[/cyan]")
        console.print(f"[dim]Thread ID: {thread_id[:8]}...[/dim]\n")
    except Exception as e:
        console.print(f"[red]Error creating thread:[/red] {e}")
        raise typer.Exit(1)
    finally:
        db.close()

    # Load config
    try:
        config = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Start chat with new thread
    console.print("[bold cyan]Local Agent - Interactive Chat[/bold cyan]")
    console.print("[dim]Type 'exit', 'quit', or press Ctrl+D to quit[/dim]\n")

    try:
        runtime = AgentRuntime(config, thread_id=thread_id)
        console.print(f"[green]✓[/green] Agent initialized\n")
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        raise typer.Exit(1)

    # Run interactive REPL
    _run_interactive_repl(runtime, console)


@threads_app.command("resume")
def threads_resume(
    thread_id: str = typer.Argument(
        ..., help="Thread ID to resume (full ID or prefix)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Resume an existing conversation thread."""
    import asyncio

    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Thread
    from ..persistence.database_init import check_database_exists, init_database
    from ..runtime import AgentRuntime

    # Initialize database if it doesn't exist
    if not check_database_exists():
        console.print("[red]Error:[/red] Database not found")
        console.print("[dim]Create a thread first with 'agent threads new'[/dim]")
        raise typer.Exit(1)

    # Find thread (support partial ID)
    db = SessionLocal()
    try:
        from sqlalchemy import select

        # Try exact match first
        thread = db.get(Thread, thread_id)

        # If not found, try prefix match
        if not thread:
            stmt = select(Thread).where(Thread.id.like(f"{thread_id}%"))
            threads = db.execute(stmt).scalars().all()

            if len(threads) == 0:
                console.print(f"[red]Error:[/red] Thread not found: {thread_id}")
                console.print(
                    "\n[dim]Use 'agent threads list' to see available threads[/dim]"
                )
                raise typer.Exit(1)
            elif len(threads) > 1:
                console.print(f"[red]Error:[/red] Ambiguous thread ID: {thread_id}")
                console.print("\n[yellow]Matching threads:[/yellow]")
                for t in threads:
                    console.print(f"  - {t.id[:8]}... {t.title}")
                raise typer.Exit(1)
            else:
                thread = threads[0]

        resolved_thread_id = thread.id
        console.print(f"[green]✓[/green] Found thread: [cyan]{thread.title}[/cyan]")
        console.print(f"[dim]Thread ID: {resolved_thread_id[:8]}...[/dim]")
        console.print(f"[dim]Messages: {len(thread.messages)}[/dim]\n")
    finally:
        db.close()

    # Load config
    try:
        config = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize runtime with thread
    console.print("[bold cyan]Local Agent - Resuming Thread[/bold cyan]")
    console.print("[dim]Type 'exit', 'quit', or press Ctrl+D to quit[/dim]\n")

    try:
        runtime = AgentRuntime(config, thread_id=resolved_thread_id)
        console.print(f"[green]✓[/green] Agent initialized\n")
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        raise typer.Exit(1)

    # Show recent messages
    if runtime.message_history:
        console.print("[yellow]Recent conversation:[/yellow]")
        # Show last 6 messages (3 exchanges)
        recent_messages = runtime.message_history[-6:]
        for msg in recent_messages:
            if msg.role == "user":
                content = str(msg.content)[:100]
                console.print(f"  [bold blue]You:[/bold blue] {content}...")
            elif msg.role == "assistant":
                content = str(msg.content)[:100]
                console.print(f"  [bold green]Assistant:[/bold green] {content}...")
        console.print()

    # Run interactive REPL
    _run_interactive_repl(runtime, console)


@threads_app.command("delete")
def threads_delete(
    thread_id: str = typer.Argument(
        ..., help="Thread ID to delete (full ID or prefix)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a conversation thread and all its messages."""
    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Thread
    from ..persistence.database_init import check_database_exists
    from sqlalchemy import select

    # Check database exists
    if not check_database_exists():
        console.print("[red]Error:[/red] Database not found")
        raise typer.Exit(1)

    db = SessionLocal()
    try:
        # Try exact match first
        thread = db.get(Thread, thread_id)

        # If not found, try prefix match
        if not thread:
            stmt = select(Thread).where(Thread.id.like(f"{thread_id}%"))
            threads = db.execute(stmt).scalars().all()

            if len(threads) == 0:
                console.print(f"[red]Error:[/red] Thread not found: {thread_id}")
                raise typer.Exit(1)
            elif len(threads) > 1:
                console.print(f"[red]Error:[/red] Ambiguous thread ID: {thread_id}")
                console.print("\n[yellow]Matching threads:[/yellow]")
                for t in threads:
                    console.print(f"  - {t.id[:8]}... {t.title}")
                raise typer.Exit(1)
            else:
                thread = threads[0]

        # Confirm deletion
        if not force:
            message_count = len(thread.messages)
            console.print(f"\n[yellow]About to delete:[/yellow]")
            console.print(f"  Title: {thread.title}")
            console.print(f"  Messages: {message_count}")
            console.print(f"  ID: {thread.id[:8]}...")

            confirm = typer.confirm("\nAre you sure you want to delete this thread?")
            if not confirm:
                console.print("[dim]Deletion cancelled[/dim]")
                raise typer.Exit(0)

        # Delete thread (cascade will delete messages)
        db.delete(thread)
        db.commit()
        console.print(f"\n[green]✓[/green] Thread deleted: {thread.title}")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error deleting thread:[/red] {e}")
        raise typer.Exit(1)
    finally:
        db.close()


# RAG command group
