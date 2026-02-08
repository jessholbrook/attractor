"""StartHandler: marks pipeline entry and records timestamp."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class StartHandler:
    """Handler for start (Mdiamond) nodes.

    Simply returns SUCCESS and sets the 'started_at' timestamp in context.
    """

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        now = datetime.now(timezone.utc).isoformat()
        return Outcome(
            status=Status.SUCCESS,
            context_updates={"started_at": now},
        )
