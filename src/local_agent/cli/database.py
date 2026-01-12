"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


db_app = typer.Typer(help="Manage local database")
# app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init() -> None:
    """Initialize database tables.

    Creates all required tables (threads, messages, sessions, message_meta).
    Safe to run multiple times - will not drop existing tables.
    """
    from ..persistence.database_init import (
        init_database,
        check_database_exists,
        get_database_info,
    )

    console.print("[bold cyan]Initializing Database[/bold cyan]\n")

    # Check if database already exists
    if check_database_exists():
        console.print("[yellow]Database already exists[/yellow]")
        info = get_database_info()
        console.print(f"[dim]Location: {info['database_url']}[/dim]")
        console.print(f"[dim]Tables: {', '.join(info['tables'])}[/dim]\n")

        confirm = typer.confirm("Initialize anyway? (will create missing tables)")
        if not confirm:
            console.print("[dim]Initialization cancelled[/dim]")
            raise typer.Exit(0)

    try:
        init_database()
        info = get_database_info()

        console.print("[green]✓[/green] Database initialized successfully\n")
        console.print("[yellow]Database Info:[/yellow]")
        console.print(f"  Location: {info['database_url']}")
        console.print(f"  Tables created: {', '.join(info['tables'])}")
    except Exception as e:
        console.print(f"[red]Error initializing database:[/red] {e}")
        raise typer.Exit(1)


@db_app.command("reset")
def db_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Reset database (DESTRUCTIVE - deletes all data!).

    Drops all tables and recreates them. All threads, messages, and sessions will be lost.
    Use with caution - this operation cannot be undone.
    """
    from ..persistence.database_init import reset_database, get_database_info

    console.print("[bold red]WARNING: Database Reset[/bold red]\n")

    if not force:
        console.print("[yellow]This will:[/yellow]")
        console.print("  • Delete ALL conversation threads")
        console.print("  • Delete ALL messages")
        console.print("  • Delete ALL sessions")
        console.print("  • Delete ALL message metadata")
        console.print("\n[red]This operation cannot be undone![/red]\n")

        confirm = typer.confirm(
            "Are you absolutely sure you want to reset the database?"
        )
        if not confirm:
            console.print("[dim]Reset cancelled[/dim]")
            raise typer.Exit(0)

        # Double confirmation
        double_confirm = typer.confirm("Type 'yes' to confirm again", default=False)
        if not double_confirm:
            console.print("[dim]Reset cancelled[/dim]")
            raise typer.Exit(0)

    try:
        console.print("\n[yellow]Resetting database...[/yellow]")
        reset_database()
        info = get_database_info()

        console.print("[green]✓[/green] Database reset complete\n")
        console.print("[yellow]Database Info:[/yellow]")
        console.print(f"  Location: {info['database_url']}")
        console.print(f"  Tables created: {', '.join(info['tables'])}")
    except Exception as e:
        console.print(f"[red]Error resetting database:[/red] {e}")
        raise typer.Exit(1)


@db_app.command("info")
def db_info() -> None:
    """Show database statistics and information."""
    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Thread, Message, Session, MessageMeta
    from ..persistence.database_init import check_database_exists, get_database_info
    from sqlalchemy import func

    if not check_database_exists():
        console.print("[red]Error:[/red] Database not found")
        console.print("\n[dim]Run 'agent db init' to create the database[/dim]")
        raise typer.Exit(1)

    db = SessionLocal()
    try:
        # Get table info
        info = get_database_info()

        # Get counts
        thread_count = db.query(func.count(Thread.id)).scalar()
        message_count = db.query(func.count(Message.id)).scalar()
        session_count = db.query(func.count(Session.id)).scalar()
        meta_count = db.query(func.count(MessageMeta.id)).scalar()

        # Get session statistics
        active_sessions = (
            db.query(func.count(Session.id)).filter(Session.status == "active").scalar()
        )
        closed_sessions = (
            db.query(func.count(Session.id)).filter(Session.status == "closed").scalar()
        )

        console.print("[bold cyan]Database Statistics[/bold cyan]\n")

        console.print("[yellow]Location:[/yellow]")
        console.print(f"  {info['database_url']}\n")

        console.print("[yellow]Tables:[/yellow]")
        console.print(f"  {', '.join(info['tables'])}\n")

        console.print("[yellow]Record Counts:[/yellow]")
        console.print(f"  Threads:         {thread_count:,}")
        console.print(f"  Messages:        {message_count:,}")
        console.print(f"  Sessions:        {session_count:,}")
        console.print(f"  Message Meta:    {meta_count:,}\n")

        console.print("[yellow]Session Status:[/yellow]")
        console.print(f"  Active:          {active_sessions:,}")
        console.print(f"  Closed:          {closed_sessions:,}")

        if thread_count > 0:
            # Get most recent thread
            from sqlalchemy import select

            stmt = select(Thread).order_by(Thread.updated_at.desc()).limit(1)
            recent_thread = db.execute(stmt).scalar_one_or_none()

            if recent_thread:
                console.print(f"\n[yellow]Most Recent Thread:[/yellow]")
                console.print(f"  Title:           {recent_thread.title}")
                console.print(f"  Messages:        {len(recent_thread.messages)}")
                console.print(
                    f"  Updated:         {recent_thread.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )

    except Exception as e:
        console.print(f"[red]Error retrieving database info:[/red] {e}")
        raise typer.Exit(1)
    finally:
        db.close()


# Providers command group
