"""ExitHandler: marks pipeline exit."""

from __future__ import annotations

from pathlib import Path

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class ExitHandler:
    """Handler for exit (Msquare) nodes.

    Returns SUCCESS -- the engine handles goal gate checking separately.
    """

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        return Outcome(status=Status.SUCCESS)
