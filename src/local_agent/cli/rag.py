"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


rag_app = typer.Typer(help="RAG management utilities")
# app.add_typer(rag_app, name="rag")


@rag_app.command("query")
def rag_query(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    score_threshold: float = typer.Option(
        0.0, "--threshold", "-t", help="Minimum similarity score (0.0-1.0)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Test RAG retrieval with a query (debug/development tool).

    Examples:
        agent rag query "How does authentication work?"
        agent rag query "database schema" --limit 3
    """
    import asyncio
    from ..services.embedding import EmbeddingService
    from ..connectors.qdrant import QdrantConnector

    console.print(f"[bold cyan]Searching:[/bold cyan] {query}\n")

    try:
        config = load_config(config_file)
        embedding_service = EmbeddingService(config.embedding)
        qdrant_connector = QdrantConnector(config.qdrant)

        async def run_search():
            query_vector = await embedding_service.embed_text(query)
            return qdrant_connector.search(
                query_vector=query_vector, limit=limit, score_threshold=score_threshold
            )

        with console.status("[cyan]Searching...[/cyan]", spinner="dots"):
            results = asyncio.run(run_search())

        if not results:
            console.print("[yellow]No results found[/yellow]")
            console.print("\n[dim]Try ingesting documents first:[/dim]")
            console.print("[dim]  agent ingest <path>[/dim]")
            return

        console.print(f"[green]✓[/green] Found {len(results)} result(s)\n")

        for i, result in enumerate(results, 1):
            payload = result.get("payload", {})
            score = result.get("score", 0.0)

            console.print(f"[bold cyan]Result {i}[/bold cyan] (score: {score:.3f})")
            console.print(f"[dim]Source:[/dim] {payload.get('source', 'unknown')}")
            console.print(f"[dim]Chunk:[/dim] {payload.get('chunk_index', 0)}")
            console.print(f"\n{payload.get('text', '')[:300]}...")
            console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@rag_app.command("list")
def rag_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Number to show"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List ingested documents."""
    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Document
    from sqlalchemy import select

    db = SessionLocal()
    try:
        stmt = select(Document).order_by(Document.ingested_at.desc()).limit(limit)
        documents = db.execute(stmt).scalars().all()

        if not documents:
            console.print("[yellow]No documents ingested yet[/yellow]")
            console.print("\n[dim]Use 'agent ingest <path>' to add documents[/dim]")
            return

        console.print(f"[bold cyan]Ingested Documents[/bold cyan] ({len(documents)})\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Source Path", style="cyan", no_wrap=False)
        table.add_column("Chunks", justify="right", style="yellow", width=8)
        table.add_column("Tokens", justify="right", style="dim", width=10)
        table.add_column("Ingested", style="dim", width=16)

        for doc in documents:
            source_display = doc.source_path
            if len(source_display) > 60:
                source_display = "..." + source_display[-57:]

            table.add_row(
                source_display,
                str(doc.chunk_count),
                f"{doc.token_count:,}",
                doc.ingested_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    finally:
        db.close()


@rag_app.command("delete")
def rag_delete(
    source_path: str = typer.Argument(
        ..., help="Source path to delete (full or partial match)"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Delete a document from the RAG store."""
    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Document
    from ..connectors.qdrant import QdrantConnector
    from sqlalchemy import select

    db = SessionLocal()
    try:
        # Find matching documents
        stmt = select(Document).where(Document.source_path.like(f"%{source_path}%"))
        documents = db.execute(stmt).scalars().all()

        if not documents:
            console.print(
                f"[red]Error:[/red] No documents found matching: {source_path}"
            )
            raise typer.Exit(1)

        if len(documents) > 1:
            console.print(
                f"[red]Error:[/red] Multiple documents match. Be more specific:"
            )
            for doc in documents:
                console.print(f"  - {doc.source_path}")
            raise typer.Exit(1)

        doc = documents[0]

        # Confirm deletion
        if not force:
            console.print(f"\n[yellow]About to delete:[/yellow]")
            console.print(f"  Source: {doc.source_path}")
            console.print(f"  Chunks: {doc.chunk_count}")
            console.print(f"  Tokens: {doc.token_count:,}")

            confirm = typer.confirm("\nAre you sure?")
            if not confirm:
                console.print("[dim]Deletion cancelled[/dim]")
                raise typer.Exit(0)

        # Delete from Qdrant
        config = load_config(config_file)
        qdrant_connector = QdrantConnector(config.qdrant)
        qdrant_connector.delete_by_source(doc.source_path)

        # Delete from database
        db.delete(doc)
        db.commit()

        console.print(f"\n[green]✓[/green] Deleted {doc.source_path}")
        console.print(f"  Removed {doc.chunk_count} chunk(s) from vector store")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        db.close()


@rag_app.command("info")
def rag_info(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Show RAG system statistics."""
    from ..persistence.db import SessionLocal
    from ..persistence.db_models import Document
    from ..connectors.qdrant import QdrantConnector
    from sqlalchemy import func

    db = SessionLocal()
    try:
        # Database stats
        doc_count = db.query(func.count(Document.id)).scalar() or 0
        total_chunks = db.query(func.sum(Document.chunk_count)).scalar() or 0
        total_tokens = db.query(func.sum(Document.token_count)).scalar() or 0

        # Qdrant stats
        config = load_config(config_file)
        qdrant_connector = QdrantConnector(config.qdrant)
        collection_info = qdrant_connector.get_collection_info()

        console.print("[bold cyan]RAG System Information[/bold cyan]\n")

        console.print("[yellow]Configuration:[/yellow]")
        console.print(f"  Qdrant URL:         {config.qdrant.url}")
        console.print(f"  Collection:         {config.qdrant.collection_name}")
        console.print(f"  Embedding Model:    {config.embedding.model}")
        console.print(f"  Vector Size:        {config.qdrant.vector_size}")
        console.print(f"  Chunk Size:         {config.rag.chunk_size} tokens")
        console.print(f"  Chunk Overlap:      {config.rag.chunk_overlap} tokens")

        console.print(f"\n[yellow]Database:[/yellow]")
        console.print(f"  Documents:          {doc_count:,}")
        console.print(f"  Total Chunks:       {total_chunks:,}")
        console.print(f"  Total Tokens:       {total_tokens:,}")

        console.print(f"\n[yellow]Vector Store:[/yellow]")
        console.print(
            f"  Points Count:       {collection_info.get('points_count', 0):,}"
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        db.close()


# Database command group
