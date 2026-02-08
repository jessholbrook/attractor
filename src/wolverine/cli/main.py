"""Wolverine CLI entry point."""
from __future__ import annotations

import click


@click.group()
def cli():
    """Wolverine: Self-healing software system."""
    pass


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=5000, type=int, help="Port to bind to")
@click.option("--db", default="wolverine.db", help="Database path")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
def serve(host: str, port: int, db: str, debug: bool) -> None:
    """Start the Wolverine web server."""
    from wolverine.config import WolverineConfig
    from wolverine.store.db import Database
    from wolverine.store.migrations import run_migrations
    from wolverine.web.app import create_app

    config = WolverineConfig(db_path=db, host=host, port=port)
    database = Database(config.db_path)
    database.connect()
    run_migrations(database)

    app = create_app(db=database)
    click.echo(f"Starting Wolverine on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


@cli.command()
@click.option("--title", required=True, help="Signal title")
@click.option("--body", required=True, help="Signal body")
@click.option("--kind", default="manual", help="Signal kind")
@click.option("--db", default="wolverine.db", help="Database path")
def ingest(title: str, body: str, kind: str, db: str) -> None:
    """Ingest a signal manually."""
    import uuid
    from datetime import datetime, timezone

    from wolverine.model.signal import RawSignal, SignalKind, SignalSource
    from wolverine.store.db import Database
    from wolverine.store.migrations import run_migrations
    from wolverine.store.repositories import SignalRepository

    database = Database(db)
    database.connect()
    run_migrations(database)

    signal = RawSignal(
        id=uuid.uuid4().hex[:12],
        kind=SignalKind(kind),
        source=SignalSource.CLI,
        title=title,
        body=body,
        received_at=datetime.now(timezone.utc).isoformat(),
    )

    repo = SignalRepository(database)
    repo.create(signal)
    database.commit()
    database.close()

    click.echo(f"Signal ingested: {signal.id}")


@cli.command("import-csv")
@click.option(
    "--file",
    "csv_file",
    required=True,
    type=click.Path(exists=True),
    help="CSV file to import",
)
@click.option("--db", default="wolverine.db", help="Database path")
def import_csv(csv_file: str, db: str) -> None:
    """Import signals from a CSV file."""
    from wolverine.adapters.csv_adapter import CSVAdapter
    from wolverine.store.db import Database
    from wolverine.store.migrations import run_migrations
    from wolverine.store.repositories import SignalRepository

    database = Database(db)
    database.connect()
    run_migrations(database)

    adapter = CSVAdapter(csv_file)
    signals = adapter.fetch()

    repo = SignalRepository(database)
    for signal in signals:
        repo.create(signal)
    database.commit()
    database.close()

    click.echo(f"Imported {len(signals)} signals")
