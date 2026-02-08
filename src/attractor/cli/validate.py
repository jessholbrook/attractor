"""CLI command: attractor validate -- parse and validate a DOT file."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from attractor.model.diagnostic import Severity
from attractor.parser import ParseError, parse_dot
from attractor.validation import validate as run_validate


@click.command()
@click.argument("dotfile", type=click.Path(exists=True))
def validate(dotfile: str) -> None:
    """Parse and validate a DOT pipeline file.

    Prints diagnostics (errors, warnings, info) and exits with code 0 if
    no errors are found, or code 1 if there are errors.
    """
    dot_path = Path(dotfile)

    # Parse
    try:
        source = dot_path.read_text(encoding="utf-8")
        graph = parse_dot(source)
    except ParseError as exc:
        click.echo(f"Parse error: {exc}", err=True)
        sys.exit(1)

    # Validate
    diagnostics = run_validate(graph)

    if not diagnostics:
        click.echo(f"OK: {dot_path.name} is valid (0 diagnostics)")
        sys.exit(0)

    # Print diagnostics grouped by severity
    errors = [d for d in diagnostics if d.severity is Severity.ERROR]
    warnings = [d for d in diagnostics if d.severity is Severity.WARNING]
    infos = [d for d in diagnostics if d.severity is Severity.INFO]

    for diag in diagnostics:
        click.echo(str(diag))

    click.echo()
    click.echo(
        f"Summary: {len(errors)} error(s), {len(warnings)} warning(s), {len(infos)} info"
    )

    if errors:
        sys.exit(1)
    sys.exit(0)
