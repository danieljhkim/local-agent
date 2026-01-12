"""CLI commands - auto-generated module."""

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from . import console, _run_interactive_repl
from ..config.loader import load_config, save_config


logs_app = typer.Typer(help="View and manage audit logs")
# app.add_typer(logs_app, name="logs")


@logs_app.command("list")
def logs_list(
    limit: int = typer.Option(
        50, "--limit", "-n", help="Maximum number of entries to show"
    ),
    event_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by event type"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s", help="Filter by session ID (prefix)"
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Show logs since date (YYYY-MM-DD)"
    ),
    until: Optional[str] = typer.Option(
        None, "--until", help="Show logs until date (YYYY-MM-DD)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """List audit log entries with optional filtering."""
    import json
    from datetime import datetime

    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    log_dir = Path(cfg.audit.log_dir).expanduser()
    if not log_dir.exists():
        console.print(f"[yellow]No logs found at:[/yellow] {log_dir}")
        console.print("[dim]Logs will be created when you use the agent[/dim]")
        return

    # Parse date filters
    since_dt = None
    until_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            console.print(
                f"[red]Error:[/red] Invalid date format for --since. Use YYYY-MM-DD"
            )
            raise typer.Exit(1)
    if until:
        try:
            until_dt = datetime.fromisoformat(until)
            until_dt = until_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            console.print(
                f"[red]Error:[/red] Invalid date format for --until. Use YYYY-MM-DD"
            )
            raise typer.Exit(1)

    # Collect log entries
    entries = []
    log_files = sorted(log_dir.glob("session_*.jsonl"), reverse=True)

    for log_file in log_files:
        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        # Apply filters
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if session and not entry.get("session_id", "").startswith(
                            session
                        ):
                            continue

                        # Date filtering
                        timestamp = datetime.fromisoformat(entry["timestamp"])
                        if since_dt and timestamp < since_dt:
                            continue
                        if until_dt and timestamp > until_dt:
                            continue

                        entries.append(entry)

                        if len(entries) >= limit:
                            break
                    except (json.JSONDecodeError, KeyError):
                        # Skip invalid entries
                        continue

            if len(entries) >= limit:
                break
        except Exception:
            continue

    if not entries:
        console.print("[yellow]No matching log entries found[/yellow]")
        if event_type or session or since or until:
            console.print("[dim]Try adjusting your filters[/dim]")
        return

    # Display results
    console.print(
        f"\n[bold cyan]Audit Log Entries[/bold cyan] [dim]({len(entries)} entries)[/dim]\n"
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="cyan", width=16)
    table.add_column("Event", style="yellow", width=15)
    table.add_column("Details", style="white")
    table.add_column("Status", style="green", width=10)

    for entry in entries:
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime(
            "%m-%d %H:%M:%S"
        )
        event_type = entry.get("event_type", "")

        # Build details string
        details = []
        if entry.get("tool_name"):
            details.append(f"[bold]{entry['tool_name']}[/bold]")
        if entry.get("thread_id"):
            details.append(f"thread:{entry['thread_id'][:8]}")
        if entry.get("session_id"):
            details.append(f"session:{entry['session_id'][:8]}")
        details_str = " | ".join(details) if details else "-"

        # Status
        status = ""
        if entry.get("success") is True:
            status = "[green]✓[/green]"
        elif entry.get("success") is False:
            status = "[red]✗[/red]"
        elif entry.get("approved") is True:
            status = "[green]✓[/green]"
        elif entry.get("approved") is False:
            status = "[red]✗[/red]"

        table.add_row(timestamp, event_type, details_str, status)

    console.print(table)


@logs_app.command("show")
def logs_show(
    session_id: str = typer.Argument(..., help="Session ID (full or prefix)"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Show detailed log entries for a specific session."""
    import json
    from datetime import datetime

    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    log_dir = Path(cfg.audit.log_dir).expanduser()
    if not log_dir.exists():
        console.print(f"[yellow]No logs found at:[/yellow] {log_dir}")
        return

    # Find matching log files
    matching_files = []
    for log_file in log_dir.glob("session_*.jsonl"):
        if session_id in log_file.name:
            matching_files.append(log_file)

    if not matching_files:
        console.print(f"[red]Error:[/red] No session found matching: {session_id}")
        return

    if len(matching_files) > 1:
        console.print(f"[red]Error:[/red] Ambiguous session ID. Multiple matches:")
        for f in matching_files:
            console.print(f"  - {f.name}")
        return

    log_file = matching_files[0]

    # Read all entries
    entries = []
    try:
        with open(log_file) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        console.print(f"[red]Error reading log file:[/red] {e}")
        raise typer.Exit(1)

    if not entries:
        console.print("[yellow]No entries found in log file[/yellow]")
        return

    # Display session summary
    first_entry = entries[0]
    console.print(
        f"\n[bold cyan]Session:[/bold cyan] {first_entry.get('session_id', 'unknown')}"
    )
    if first_entry.get("thread_id"):
        console.print(f"[bold cyan]Thread:[/bold cyan] {first_entry['thread_id']}")
    console.print(
        f"[bold cyan]Started:[/bold cyan] {datetime.fromisoformat(entries[0]['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    console.print(f"[bold cyan]Entries:[/bold cyan] {len(entries)}\n")

    # Count event types
    tool_calls = sum(1 for e in entries if e.get("event_type") == "tool_call")
    approvals = sum(1 for e in entries if e.get("event_type") == "approval")
    successes = sum(
        1 for e in entries if e.get("event_type") == "tool_call" and e.get("success")
    )
    failures = sum(
        1
        for e in entries
        if e.get("event_type") == "tool_call" and not e.get("success")
    )

    console.print(
        f"[dim]Tool calls: {tool_calls} | Approvals: {approvals} | Success: {successes} | Failures: {failures}[/dim]\n"
    )

    # Display entries
    for i, entry in enumerate(entries, 1):
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
        event_type = entry.get("event_type", "unknown")

        if event_type == "tool_call":
            status = "✓" if entry.get("success") else "✗"
            color = "green" if entry.get("success") else "red"
            console.print(
                f"[{color}]{status}[/{color}] [{timestamp}] [yellow]{event_type}[/yellow] - [bold]{entry.get('tool_name')}[/bold]"
            )
            if entry.get("error"):
                console.print(f"    [red]Error:[/red] {entry['error']}")
        elif event_type == "approval":
            status = "✓" if entry.get("approved") else "✗"
            color = "green" if entry.get("approved") else "red"
            console.print(
                f"[{color}]{status}[/{color}] [{timestamp}] [yellow]{event_type}[/yellow] - {entry.get('tool_name')}"
            )
        else:
            console.print(f"  [{timestamp}] [yellow]{event_type}[/yellow]")


@logs_app.command("tail")
def logs_tail(
    lines: int = typer.Option(20, "--lines", "-n", help="Number of lines to show"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Show the most recent log entries across all sessions."""
    import json
    from datetime import datetime

    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    log_dir = Path(cfg.audit.log_dir).expanduser()
    if not log_dir.exists():
        console.print(f"[yellow]No logs found at:[/yellow] {log_dir}")
        return

    # Collect recent entries
    entries = []
    log_files = sorted(log_dir.glob("session_*.jsonl"), reverse=True)

    for log_file in log_files[:10]:  # Check last 10 session files
        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry["_file"] = log_file.name
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    # Sort by timestamp and take last N
    entries.sort(key=lambda e: e["timestamp"])
    recent = entries[-lines:]

    if not recent:
        console.print("[yellow]No log entries found[/yellow]")
        return

    console.print(
        f"\n[bold cyan]Recent Log Entries[/bold cyan] [dim](last {len(recent)})[/dim]\n"
    )

    for entry in recent:
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime(
            "%m-%d %H:%M:%S"
        )
        event_type = entry.get("event_type", "")

        if event_type == "tool_call":
            status = "✓" if entry.get("success") else "✗"
            color = "green" if entry.get("success") else "red"
            console.print(
                f"[{color}]{status}[/{color}] [{timestamp}] {entry.get('tool_name')}"
            )
        elif event_type == "approval":
            status = "✓" if entry.get("approved") else "✗"
            color = "green" if entry.get("approved") else "red"
            console.print(
                f"[{color}]{status}[/{color}] [{timestamp}] approval: {entry.get('tool_name')}"
            )
        else:
            console.print(f"  [{timestamp}] {event_type}")


@logs_app.command("export")
def logs_export(
    output: Path = typer.Argument(..., help="Output file path (.json or .csv)"),
    event_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by event type"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s", help="Filter by session ID (prefix)"
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Export logs since date (YYYY-MM-DD)"
    ),
    until: Optional[str] = typer.Option(
        None, "--until", help="Export logs until date (YYYY-MM-DD)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Export audit logs to JSON or CSV format."""
    import json
    import csv
    from datetime import datetime

    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    log_dir = Path(cfg.audit.log_dir).expanduser()
    if not log_dir.exists():
        console.print(f"[yellow]No logs found at:[/yellow] {log_dir}")
        return

    # Parse date filters
    since_dt = None
    until_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid date format for --since")
            raise typer.Exit(1)
    if until:
        try:
            until_dt = datetime.fromisoformat(until)
            until_dt = until_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid date format for --until")
            raise typer.Exit(1)

    # Collect entries
    entries = []
    for log_file in sorted(log_dir.glob("session_*.jsonl")):
        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        # Apply filters
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if session and not entry.get("session_id", "").startswith(
                            session
                        ):
                            continue

                        timestamp = datetime.fromisoformat(entry["timestamp"])
                        if since_dt and timestamp < since_dt:
                            continue
                        if until_dt and timestamp > until_dt:
                            continue

                        entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            continue

    if not entries:
        console.print("[yellow]No matching entries to export[/yellow]")
        return

    # Export
    try:
        if output.suffix == ".json":
            with open(output, "w") as f:
                json.dump(entries, f, indent=2)
        elif output.suffix == ".csv":
            # Get all unique keys
            all_keys = set()
            for entry in entries:
                all_keys.update(entry.keys())
            fieldnames = sorted(all_keys)

            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in entries:
                    # Convert non-string values to strings for CSV
                    row = {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in entry.items()
                    }
                    writer.writerow(row)
        else:
            console.print(
                f"[red]Error:[/red] Unsupported file format. Use .json or .csv"
            )
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Exported {len(entries)} entries to: {output}")
    except Exception as e:
        console.print(f"[red]Error exporting:[/red] {e}")
        raise typer.Exit(1)


@logs_app.command("stats")
def logs_stats(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Display statistics about audit logs."""
    import json
    from datetime import datetime
    from collections import Counter

    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    log_dir = Path(cfg.audit.log_dir).expanduser()
    if not log_dir.exists():
        console.print(f"[yellow]No logs found at:[/yellow] {log_dir}")
        return

    log_files = list(log_dir.glob("session_*.jsonl"))
    if not log_files:
        console.print("[yellow]No log files found[/yellow]")
        return

    # Collect statistics
    total_entries = 0
    event_types = Counter()
    tool_calls = Counter()
    sessions = set()
    threads = set()
    successes = 0
    failures = 0
    approvals_granted = 0
    approvals_denied = 0
    earliest = None
    latest = None

    for log_file in log_files:
        try:
            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        total_entries += 1

                        event_types[entry.get("event_type", "unknown")] += 1

                        if entry.get("session_id"):
                            sessions.add(entry["session_id"])
                        if entry.get("thread_id"):
                            threads.add(entry["thread_id"])

                        if entry.get("event_type") == "tool_call":
                            tool_calls[entry.get("tool_name", "unknown")] += 1
                            if entry.get("success"):
                                successes += 1
                            else:
                                failures += 1

                        if entry.get("event_type") == "approval":
                            if entry.get("approved"):
                                approvals_granted += 1
                            else:
                                approvals_denied += 1

                        timestamp = datetime.fromisoformat(entry["timestamp"])
                        if earliest is None or timestamp < earliest:
                            earliest = timestamp
                        if latest is None or timestamp > latest:
                            latest = timestamp
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            continue

    # Display statistics
    console.print("\n[bold cyan]Audit Log Statistics[/bold cyan]\n")

    console.print(f"[bold]Overview[/bold]")
    console.print(f"  Total log files: {len(log_files)}")
    console.print(f"  Total entries: {total_entries}")
    console.print(f"  Unique sessions: {len(sessions)}")
    console.print(f"  Unique threads: {len(threads)}")
    if earliest and latest:
        console.print(
            f"  Date range: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
        )
    console.print()

    console.print(f"[bold]Event Types[/bold]")
    for event, count in event_types.most_common():
        console.print(f"  {event}: {count}")
    console.print()

    if tool_calls:
        console.print(f"[bold]Top Tools[/bold]")
        for tool, count in tool_calls.most_common(10):
            console.print(f"  {tool}: {count}")
        console.print()

    if successes + failures > 0:
        total = successes + failures
        success_rate = (successes / total) * 100
        console.print(f"[bold]Tool Call Success Rate[/bold]")
        console.print(f"  Successes: [green]{successes}[/green]")
        console.print(f"  Failures: [red]{failures}[/red]")
        console.print(f"  Success rate: {success_rate:.1f}%")
        console.print()

    if approvals_granted + approvals_denied > 0:
        console.print(f"[bold]Approvals[/bold]")
        console.print(f"  Granted: [green]{approvals_granted}[/green]")
        console.print(f"  Denied: [red]{approvals_denied}[/red]")


@logs_app.command("clean")
def logs_clean(
    older_than: int = typer.Option(
        30, "--older-than", help="Remove logs older than N days"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
) -> None:
    """Remove old audit log files."""
    import json
    from datetime import datetime, timedelta

    try:
        cfg = load_config(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

    log_dir = Path(cfg.audit.log_dir).expanduser()
    if not log_dir.exists():
        console.print(f"[yellow]No logs found at:[/yellow] {log_dir}")
        return

    cutoff_date = datetime.now() - timedelta(days=older_than)

    # Find old log files
    old_files = []
    for log_file in log_dir.glob("session_*.jsonl"):
        try:
            # Parse timestamp from filename: session_20240110_103000_abc123.jsonl
            parts = log_file.stem.split("_")
            if len(parts) >= 3:
                file_date_str = f"{parts[1]}_{parts[2]}"
                file_date = datetime.strptime(file_date_str, "%Y%m%d_%H%M%S")

                if file_date < cutoff_date:
                    old_files.append((log_file, file_date))
        except Exception:
            continue

    if not old_files:
        console.print(
            f"[yellow]No log files older than {older_than} days found[/yellow]"
        )
        return

    # Show files to be deleted
    console.print(
        f"\n[bold yellow]Found {len(old_files)} log files older than {older_than} days:[/bold yellow]\n"
    )
    for log_file, file_date in old_files[:10]:
        console.print(
            f"  - {log_file.name} [dim]({file_date.strftime('%Y-%m-%d')})[/dim]"
        )

    if len(old_files) > 10:
        console.print(f"  [dim]... and {len(old_files) - 10} more[/dim]")

    # Confirmation
    if not force:
        confirm = typer.confirm("\nDelete these files?", default=False)
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    # Delete files
    deleted = 0
    for log_file, _ in old_files:
        try:
            log_file.unlink()
            deleted += 1
        except Exception as e:
            console.print(f"[red]Error deleting {log_file.name}:[/red] {e}")

    console.print(f"\n[green]✓[/green] Deleted {deleted} log files")


# Memory command group
