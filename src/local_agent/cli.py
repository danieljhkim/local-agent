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
    thread_id: Optional[str] = typer.Option(
        None, "--thread", "-t", help="Thread ID to resume (or 'new' to create)"
    ),
) -> None:
    """Start an interactive chat session with the agent.

    By default, starts an ephemeral session (not saved to database).
    Use --thread to resume a specific thread or --thread=new to create a new one.
    """
    import asyncio
    import uuid

    from .runtime import AgentRuntime

    console.print("[bold cyan]Local Agent - Interactive Chat[/bold cyan]")
    console.print("[dim]Type 'exit', 'quit', or press Ctrl+D to quit[/dim]\n")

    # Handle thread creation/resumption
    resolved_thread_id = None
    if thread_id == "new":
        # Create new thread
        from .persistence.db import SessionLocal
        from .persistence.db_models import Thread
        from .persistence.init import check_database_exists, init_database

        if not check_database_exists():
            console.print("[yellow]Database not found. Initializing...[/yellow]")
            init_database()

        resolved_thread_id = str(uuid.uuid4())
        db = SessionLocal()
        try:
            thread = Thread(id=resolved_thread_id, title="Chat session")
            db.add(thread)
            db.commit()
            console.print(f"[green]✓[/green] Created new thread: {resolved_thread_id[:8]}...")
        except Exception as e:
            console.print(f"[red]Error creating thread:[/red] {e}")
            raise typer.Exit(1)
        finally:
            db.close()
    elif thread_id:
        # Resume existing thread
        resolved_thread_id = thread_id
        console.print(f"[green]✓[/green] Resuming thread: {thread_id[:8]}...")

    # Load config
    try:
        config = load_config(config_file)
        console.print(f"[green]✓[/green] Configuration loaded")
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize runtime
    try:
        runtime = AgentRuntime(config, thread_id=resolved_thread_id)
        console.print(f"[green]✓[/green] Agent initialized")
        if not resolved_thread_id:
            console.print(f"[dim]Note: Running in ephemeral mode (not saved)[/dim]")
        console.print()
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        raise typer.Exit(1)

    # Create persistent event loop for the session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # REPL loop
    try:
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

                # Execute task using persistent event loop
                response = loop.run_until_complete(runtime.execute(user_input))

                # Note: Response is already printed by runtime.execute()
                # Only print if we got a response but nothing was displayed
                # (this shouldn't happen in normal operation)

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
    finally:
        # Always shutdown runtime to close database session
        runtime.shutdown()


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
    finally:
        runtime.shutdown()


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


# Threads command group
threads_app = typer.Typer(help="Manage conversation threads")
app.add_typer(threads_app, name="threads")


@threads_app.command("list")
def threads_list(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of threads to show"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List recent conversation threads."""
    from .persistence.db import SessionLocal
    from .persistence.db_models import Thread
    from .persistence.init import check_database_exists, init_database
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

        console.print(f"[bold cyan]Recent Threads[/bold cyan] (showing {len(threads)} of {limit})\n")

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
        console.print(f"\n[dim]Use 'agent threads resume <id>' to continue a conversation[/dim]")
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

    from .persistence.db import SessionLocal
    from .persistence.db_models import Thread
    from .persistence.init import check_database_exists, init_database
    from .runtime import AgentRuntime

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

    # Create persistent event loop for the session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # REPL loop
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                runtime.shutdown()
                break

            if not user_input.strip():
                continue

            response = loop.run_until_complete(runtime.execute(user_input))

        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            runtime.shutdown()
            break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]You can continue or type 'exit' to quit[/dim]")


@threads_app.command("resume")
def threads_resume(
    thread_id: str = typer.Argument(..., help="Thread ID to resume (full ID or prefix)"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Resume an existing conversation thread."""
    import asyncio

    from .persistence.db import SessionLocal
    from .persistence.db_models import Thread
    from .persistence.init import check_database_exists, init_database
    from .runtime import AgentRuntime

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
                console.print("\n[dim]Use 'agent threads list' to see available threads[/dim]")
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

    # Create persistent event loop for the session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # REPL loop
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                runtime.shutdown()
                break

            if not user_input.strip():
                continue

            response = loop.run_until_complete(runtime.execute(user_input))

        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            runtime.shutdown()
            break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]You can continue or type 'exit' to quit[/dim]")


@threads_app.command("delete")
def threads_delete(
    thread_id: str = typer.Argument(..., help="Thread ID to delete (full ID or prefix)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a conversation thread and all its messages."""
    from .persistence.db import SessionLocal
    from .persistence.db_models import Thread
    from .persistence.init import check_database_exists
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


# Providers command group
providers_app = typer.Typer(help="Manage LLM providers")
app.add_typer(providers_app, name="providers")


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
        console.print("\n[dim]Add providers to your config file or use 'agent config --init'[/dim]")
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
    console.print(f"\n[dim]Default provider: [cyan]{cfg.default_provider or 'None'}[/cyan][/dim]")


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
            Path.home() / ".config" / "agent" / "config.yaml",
            Path.home() / ".local" / "agent" / "config.yaml",
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
        console.print(f"[green]✓[/green] Default provider set to: [cyan]{provider_name}[/cyan]")
        console.print(f"[dim]Config saved to: {config_file}[/dim]")
    except Exception as e:
        console.print(f"[red]Error saving config:[/red] {e}")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
