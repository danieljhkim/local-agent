"""CLI entry point for the local agent."""

import asyncio

import typer
from rich.console import Console

from .. import __version__

# Shared console instance
console = Console()


def _run_interactive_repl(runtime, console: Console) -> None:
    """Run an interactive REPL loop with the agent.

    Handles user input, exit commands, and keyboard interrupts.
    Uses a persistent event loop for the session.

    Args:
        runtime: AgentRuntime instance to execute commands
        console: Rich console for output
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ")

                if user_input.strip().lower() in ["exit", "quit"]:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not user_input.strip():
                    continue

                loop.run_until_complete(runtime.execute(user_input))

            except EOFError:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
                console.print("[dim]You can continue or type 'exit' to quit[/dim]")
    finally:
        runtime.shutdown()


# Create main app
app = typer.Typer(
    name="agent",
    help="Local AI agent with tool support and safety-first design",
    add_completion=False,
)


# Import and register command modules
from . import database, identity, logs, main, memory, providers, rag, threads

# Register command groups
app.add_typer(threads.threads_app, name="threads")
app.add_typer(rag.rag_app, name="rag")
app.add_typer(database.db_app, name="db")
app.add_typer(providers.providers_app, name="providers")
app.add_typer(logs.logs_app, name="logs")
app.add_typer(memory.memory_app, name="memory")
app.add_typer(identity.identity_app, name="identity")

# Register main commands
app.command()(main.chat)
app.command()(main.run)
app.command()(main.tools)
app.command()(main.config)
app.command()(main.version)
app.command()(main.ingest)


def cli_main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    cli_main()
