"""ParallelHandler: executes child nodes concurrently via a thread pool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status

if TYPE_CHECKING:
    from attractor.engine.engine import Handler


class ParallelHandler:
    """Handler for parallel (component) nodes.

    The node's prompt lists child node IDs (comma-separated).
    Uses a ThreadPoolExecutor to execute their handlers concurrently.
    Merges context updates from all children.

    Returns:
    - SUCCESS if all children succeed
    - PARTIAL_SUCCESS if some succeed
    - FAIL if all fail
    """

    def __init__(self, registry: object | None = None) -> None:
        """Accept an optional handler registry for resolving child handlers."""
        self._registry = registry

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        prompt = node.prompt or node.label
        child_ids = [cid.strip() for cid in prompt.split(",") if cid.strip()]

        if not child_ids:
            return Outcome(status=Status.SUCCESS, notes="No child nodes specified")

        # Resolve child nodes
        children: list[Node] = []
        for cid in child_ids:
            if cid in graph.nodes:
                children.append(graph.nodes[cid])

        if not children:
            return Outcome(
                status=Status.FAIL,
                failure_reason=f"No valid child nodes found: {child_ids}",
            )

        # Execute children in parallel
        outcomes: list[Outcome] = []
        merged_updates: dict = {}

        def _run_child(child: Node) -> Outcome:
            if self._registry is not None:
                handler = self._registry.resolve(child)  # type: ignore[attr-defined]
                return handler.execute(child, context, graph, logs_root / child.id)
            # Without a registry, mark as success (for simple testing scenarios)
            return Outcome(status=Status.SUCCESS)

        with ThreadPoolExecutor(max_workers=len(children)) as pool:
            futures = {pool.submit(_run_child, child): child for child in children}
            for future in as_completed(futures):
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = Outcome(status=Status.FAIL, failure_reason=str(exc))
                outcomes.append(outcome)
                if outcome.context_updates:
                    merged_updates.update(outcome.context_updates)

        # Determine aggregate status
        successes = sum(1 for o in outcomes if o.succeeded)
        total = len(outcomes)

        if successes == total:
            agg_status = Status.SUCCESS
        elif successes > 0:
            agg_status = Status.PARTIAL_SUCCESS
        else:
            agg_status = Status.FAIL

        # Mark this parallel node as complete in context
        merged_updates[f"{node.id}.complete"] = True

        return Outcome(
            status=agg_status,
            context_updates=merged_updates,
            notes=f"{successes}/{total} children succeeded",
        )
