"""CLI command: attractor run -- execute a DOT pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from attractor.engine.engine import Engine, HandlerRegistry
from attractor.events.bus import EventBus
from attractor.model.checkpoint import Checkpoint
from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status
from attractor.parser import ParseError, parse_dot
from attractor.transforms import apply_transforms
from attractor.validation import validate_or_raise, ValidationError


# ---------------------------------------------------------------------------
# Stub / fallback handlers used when real handlers are not yet available
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal backend that always returns SUCCESS -- used for dry-run / testing."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        return Outcome(status=Status.SUCCESS)


class _StartHandler:
    """Handler for Start (Mdiamond) nodes: always succeeds."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        return Outcome(status=Status.SUCCESS)


class _ExitHandler:
    """Handler for Exit (Msquare) nodes: always succeeds."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        return Outcome(status=Status.SUCCESS)


class _AutoApproveHandler:
    """Handler for wait.human nodes when --auto-approve is set."""

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        return Outcome(status=Status.SUCCESS, preferred_label="approve")


def _build_registry(
    backend: str = "stub",
    auto_approve: bool = False,
) -> HandlerRegistry:
    """Build a HandlerRegistry with the appropriate handlers.

    Tries to import the real handler implementations first.  If they are not
    yet available (another agent is building them), falls back to the built-in
    stubs so the CLI can function independently.
    """
    registry = HandlerRegistry()

    # Start and Exit are always the same simple handlers
    registry.register("start", _StartHandler())
    registry.register("exit", _ExitHandler())

    # Codergen / default handler
    if backend == "stub":
        codergen_handler = _StubBackend()
    else:
        # Try to import a real backend; fall back to stub
        try:
            from attractor.handlers.codergen import CodergenHandler  # type: ignore[import-not-found]
            codergen_handler = CodergenHandler(backend=backend)
        except (ImportError, ModuleNotFoundError):
            click.echo(f"Warning: backend '{backend}' not available, using stub", err=True)
            codergen_handler = _StubBackend()

    registry.register("codergen", codergen_handler)
    registry.set_default(codergen_handler)

    # Conditional (diamond) -- just uses the codergen handler
    registry.register("conditional", codergen_handler)

    # Wait/human gate
    if auto_approve:
        registry.register("wait.human", _AutoApproveHandler())
    else:
        try:
            from attractor.handlers.wait_human import WaitHumanHandler  # type: ignore[import-not-found]
            from attractor.interviewer.auto import AutoApproveInterviewer  # type: ignore[import-not-found]
            registry.register("wait.human", WaitHumanHandler())
        except (ImportError, ModuleNotFoundError):
            # No interviewer available -- use auto-approve as safe default in CLI
            registry.register("wait.human", _AutoApproveHandler())

    return registry


@click.command()
@click.argument("dotfile", type=click.Path(exists=True))
@click.option("--backend", default="stub", help="Codergen backend to use")
@click.option("--auto-approve", is_flag=True, help="Auto-approve human gates")
@click.option("--logs-dir", default=None, help="Custom run directory")
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume from checkpoint file",
)
@click.option("--goal", default=None, help="Override pipeline goal")
def run(
    dotfile: str,
    backend: str,
    auto_approve: bool,
    logs_dir: str | None,
    resume: str | None,
    goal: str | None,
) -> None:
    """Execute a DOT pipeline file.

    Parses, validates, transforms, and runs the pipeline through the engine.
    """
    dot_path = Path(dotfile)

    # Step 1: Parse
    try:
        source = dot_path.read_text(encoding="utf-8")
        graph = parse_dot(source)
    except ParseError as exc:
        click.echo(f"Parse error: {exc}", err=True)
        sys.exit(1)

    # Step 2: Validate
    try:
        warnings = validate_or_raise(graph)
        for w in warnings:
            click.echo(f"  {w}", err=True)
    except ValidationError as exc:
        click.echo(f"Validation failed: {exc}", err=True)
        sys.exit(1)

    # Step 3: Transform
    graph = apply_transforms(graph)

    # Step 4: Override goal if requested
    if goal:
        graph.attributes["goal"] = goal

    # Step 5: Build registry
    registry = _build_registry(backend=backend, auto_approve=auto_approve)

    # Step 6: Set up logs directory
    if logs_dir:
        logs_root = Path(logs_dir)
    else:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        logs_root = Path(f"attractor-runs/{graph.name}-{ts}")

    # Step 7: Load checkpoint if resuming
    checkpoint = None
    if resume:
        checkpoint = Checkpoint.load(Path(resume))
        click.echo(f"Resuming from checkpoint: {resume}")

    # Step 8: Create context and event bus
    context = Context()
    event_bus = EventBus()

    # Step 9: Run engine
    click.echo(f"Running pipeline: {graph.name}")
    if graph.goal:
        click.echo(f"Goal: {graph.goal}")
    click.echo()

    engine = Engine(
        graph,
        registry,
        context=context,
        event_bus=event_bus,
        logs_root=logs_root,
        checkpoint=checkpoint,
    )
    outcome = engine.run()

    # Step 10: Print results
    click.echo()
    click.echo(f"Outcome: {outcome.status.value}")
    if outcome.failure_reason:
        click.echo(f"Failure: {outcome.failure_reason}")
    if outcome.notes:
        click.echo(f"Notes: {outcome.notes}")

    # Summary of completed nodes
    completed = engine._completed_nodes
    if completed:
        click.echo(f"\nCompleted nodes ({len(completed)}):")
        for nid in completed:
            click.echo(f"  - {nid}")

    click.echo(f"\nRun directory: {logs_root}")

    # Exit code
    if outcome.status in (Status.SUCCESS, Status.PARTIAL_SUCCESS):
        sys.exit(0)
    else:
        sys.exit(1)
