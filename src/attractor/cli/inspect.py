"""CLI command: attractor inspect -- display graph structure."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from attractor.parser import ParseError, parse_dot


@click.command()
@click.argument("dotfile", type=click.Path(exists=True))
def inspect(dotfile: str) -> None:
    """Parse a DOT file and display its graph structure.

    Shows graph metadata, nodes (with type/shape), and edges (with conditions).
    """
    dot_path = Path(dotfile)

    try:
        source = dot_path.read_text(encoding="utf-8")
        graph = parse_dot(source)
    except ParseError as exc:
        click.echo(f"Parse error: {exc}", err=True)
        sys.exit(1)

    # Graph info
    click.echo(f"Graph: {graph.name}")
    if graph.goal:
        click.echo(f"Goal:  {graph.goal}")
    click.echo(f"Nodes: {len(graph.nodes)}")
    click.echo(f"Edges: {len(graph.edges)}")
    click.echo()

    # Nodes
    click.echo("Nodes:")
    for node in graph.nodes.values():
        parts = [f"  {node.id}"]
        if node.label and node.label != node.id:
            parts.append(f'label="{node.label}"')
        parts.append(f"shape={node.shape}")
        if node.type:
            parts.append(f"type={node.type}")
        if node.prompt:
            prompt_display = node.prompt[:50] + "..." if len(node.prompt) > 50 else node.prompt
            parts.append(f'prompt="{prompt_display}"')
        click.echo("  ".join(parts))
    click.echo()

    # Edges
    click.echo("Edges:")
    for edge in graph.edges:
        parts = [f"  {edge.from_node} -> {edge.to_node}"]
        if edge.label:
            parts.append(f'label="{edge.label}"')
        if edge.condition:
            parts.append(f"condition={edge.condition}")
        if edge.weight:
            parts.append(f"weight={edge.weight}")
        click.echo("  ".join(parts))
