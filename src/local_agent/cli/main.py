"""Core CLI commands: chat, run, tools, config, version, ingest."""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

import typer

from . import console, _run_interactive_repl
from .. import __version__
from ..config.loader import load_config, save_config
from ..config.schema import AgentConfig


def chat(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    thread_id: Optional[str] = typer.Option(
        None, "--thread", "-t", help="Thread ID to resume (or 'new' to create)"
    ),
    identity: Optional[str] = typer.Option(
        None, "--identity", "-i", help="Identity to use (overrides active identity)"
    ),
) -> None:
    """Start an interactive chat session with the agent.

    By default, starts an ephemeral session (not saved to database).
    Use --thread to resume a specific thread or --thread=new to create a new one.
    Use --identity to override the active identity for this session.
    """
    from ..runtime import AgentRuntime

    console.print("[bold cyan]Local Agent - Interactive Chat[/bold cyan]")
    console.print("[dim]Type 'exit', 'quit', or press Ctrl+D to quit[/dim]\n")

    # Resolve identity
    system_prompt = None
    if identity:
        from ..identities import get_identity_manager

        manager = get_identity_manager()
        try:
            system_prompt = manager.get_content(identity)
            console.print(f"[green]✓[/green] Using identity: [cyan]{identity}[/cyan]")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Handle thread creation/resumption
    resolved_thread_id = None
    if thread_id == "new":
        # Create new thread
        from ..persistence.db import SessionLocal
        from ..persistence.db_models import Thread
        from ..persistence.database_init import check_database_exists, init_database

        if not check_database_exists():
            console.print("[yellow]Database not found. Initializing...[/yellow]")
            init_database()

        resolved_thread_id = str(uuid.uuid4())
        db = SessionLocal()
        try:
            thread = Thread(id=resolved_thread_id, title="Chat session")
            db.add(thread)
            db.commit()
            console.print(
                f"[green]✓[/green] Created new thread: {resolved_thread_id[:8]}..."
            )
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
        runtime = AgentRuntime(
            config, thread_id=resolved_thread_id, system_prompt=system_prompt
        )
        console.print(f"[green]✓[/green] Agent initialized")
        if not resolved_thread_id:
            console.print(f"[dim]Note: Running in ephemeral mode (not saved)[/dim]")
        console.print()
    except Exception as e:
        console.print(f"[red]Error initializing agent:[/red] {e}")
        raise typer.Exit(1)

    # Run interactive REPL
    _run_interactive_repl(runtime, console)


def run(
    task: str = typer.Argument(..., help="Task description for the agent"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    identity: Optional[str] = typer.Option(
        None, "--identity", "-i", help="Identity to use (overrides active identity)"
    ),
) -> None:
    """Run a single task and exit.

    Use --identity to override the active identity for this task.
    """
    from ..runtime import AgentRuntime

    console.print(f"[bold cyan]Task:[/bold cyan] {task}\n")

    # Resolve identity
    system_prompt = None
    if identity:
        from ..identities import get_identity_manager

        manager = get_identity_manager()
        try:
            system_prompt = manager.get_content(identity)
            console.print(f"[green]✓[/green] Using identity: [cyan]{identity}[/cyan]")
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Load config
    try:
        config = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize runtime
    try:
        runtime = AgentRuntime(config, system_prompt=system_prompt)
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


def tools(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List available tools and their permissions."""
    from rich.table import Table

    from ..connectors.filesystem import FilesystemConnector
    from ..tools.filesystem import register_filesystem_tools
    from ..tools.registry import ToolRegistry

    console.print("[yellow]Available Tools[/yellow]\n")

    # Load config to get workspace settings
    try:
        config = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    # Initialize registry and register tools
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
            config_file = Path.home() / ".config" / "local-agent" / "config.yaml"

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
            console.print(
                f"\n[cyan]Approval Policies:[/cyan] {len(cfg.approval_policies)}"
            )
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


def version() -> None:
    """Show version information."""
    console.print(f"[cyan]local-agent[/cyan] version [green]{__version__}[/green]")


def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest"),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r",
        help="Recursively ingest directories",
    ),
    glob_pattern: str = typer.Option(
        "**/*", "--pattern", "-p", help="Glob pattern for files (e.g., '**/*.py')"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-ingest even if already processed"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Ingest documents into the RAG knowledge store.

    Examples:
        agent ingest ~/repos/myproject
        agent ingest ~/docs/notes.md
        agent ingest ~/code --pattern "**/*.py"
    """
    import time

    from ..connectors.qdrant import QdrantConnector
    from ..persistence.db import SessionLocal
    from ..services.embedding import EmbeddingService
    from ..services.ingestion import IngestionPipeline

    console.print(f"[bold cyan]Ingesting:[/bold cyan] {path}\n")

    db_session = None
    try:
        config = load_config(config_file)

        # Ensure database is initialized
        from ..persistence.database_init import check_database_exists, init_database

        if not check_database_exists():
            console.print("[yellow]Database not found. Initializing...[/yellow]")
            init_database()

        # Initialize components
        embedding_service = EmbeddingService(config.embedding)
        qdrant_connector = QdrantConnector(config.qdrant)
        qdrant_connector.ensure_collection()

        db_session = SessionLocal()

        pipeline = IngestionPipeline(
            embedding_service=embedding_service,
            qdrant_connector=qdrant_connector,
            rag_config=config.rag,
            db_session=db_session,
        )

        # Run ingestion
        async def run_ingestion():
            start_time = time.time()

            if Path(path).is_file():
                result = await pipeline.ingest_file(path)
                result["elapsed_seconds"] = time.time() - start_time
            else:
                result = await pipeline.ingest_directory(
                    path, glob_pattern=glob_pattern, recursive=recursive
                )
                result["elapsed_seconds"] = time.time() - start_time

            return result

        with console.status("[cyan]Processing documents...[/cyan]", spinner="dots"):
            result = asyncio.run(run_ingestion())

        # Display results
        console.print("\n[bold green]✓ Ingestion Complete[/bold green]\n")

        # Handle both single file and directory results
        if Path(path).is_file():
            # Single file result
            status = result.get("status", "unknown")
            if status == "success":
                console.print(
                    f"[cyan]Chunks created:[/cyan] {result['chunks_created']}"
                )
                console.print(f"[cyan]Total tokens:[/cyan] {result['total_tokens']}")
            elif status == "skipped":
                console.print(
                    f"[yellow]Skipped:[/yellow] {result.get('reason')} - {result.get('message')}"
                )
            elif status == "error":
                console.print(f"[red]Error:[/red] {result.get('error')}")
        else:
            # Directory result
            console.print(f"[cyan]Files processed:[/cyan] {result['files_processed']}")
            console.print(f"[cyan]Files ingested:[/cyan] {result['files_ingested']}")
            if result.get("files_skipped"):
                console.print(
                    f"[yellow]Files skipped:[/yellow] {result['files_skipped']}"
                )
            if result.get("files_errored"):
                console.print(f"[red]Files errored:[/red] {result['files_errored']}")
            console.print(f"[cyan]Chunks created:[/cyan] {result['chunks_created']}")
            console.print(f"[cyan]Total tokens:[/cyan] {result['total_tokens']}")

            # Show error sample if present
            if result.get("error_sample"):
                console.print(f"\n[yellow]Sample error:[/yellow]")
                console.print(f"  [dim]File:[/dim] {result['error_sample']['file']}")
                console.print(f"  [dim]Error:[/dim] {result['error_sample']['error']}")

        console.print(f"[cyan]Elapsed time:[/cyan] {result['elapsed_seconds']:.2f}s")

    except Exception as e:
        console.print(f"\n[red]Error during ingestion:[/red] {e}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)
    finally:
        if db_session:
            db_session.close()
