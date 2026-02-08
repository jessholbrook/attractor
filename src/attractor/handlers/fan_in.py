"""FanInHandler: synchronization point for parallel branches."""

from __future__ import annotations

from pathlib import Path

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class FanInHandler:
    """Handler for parallel.fan_in (tripleoctagon) nodes.

    Waits for all expected predecessor nodes to have completed by
    checking context for completion markers from parallel branches.
    Predecessors are identified from incoming edges to this node.

    Returns SUCCESS when all predecessors are done.
    """

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        incoming = graph.incoming_edges(node.id)

        if not incoming:
            return Outcome(status=Status.SUCCESS, notes="No predecessors to wait for")

        predecessor_ids = [edge.from_node for edge in incoming]
        missing: list[str] = []

        for pred_id in predecessor_ids:
            # Check for completion marker: either '{pred_id}.complete' in context
            # or the predecessor appears in a parallel completion set.
            marker = context.get(f"{pred_id}.complete")
            if not marker:
                missing.append(pred_id)

        if missing:
            return Outcome(
                status=Status.RETRY,
                notes=f"Waiting for predecessors: {', '.join(missing)}",
            )

        return Outcome(
            status=Status.SUCCESS,
            notes=f"All {len(predecessor_ids)} predecessors completed",
        )
