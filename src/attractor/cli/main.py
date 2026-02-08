"""Attractor CLI entry point: Click group with subcommands."""

import click

from attractor import __version__


@click.group()
@click.version_option(version=__version__, prog_name="attractor")
def cli() -> None:
    """Attractor - DOT-based pipeline runner for multi-stage AI workflows."""


# Import and register subcommands
from attractor.cli.run import run  # noqa: E402
from attractor.cli.validate import validate  # noqa: E402
from attractor.cli.inspect import inspect  # noqa: E402

cli.add_command(run)
cli.add_command(validate)
cli.add_command(inspect)
