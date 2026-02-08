"""StackManagerHandler: manages a sequential loop over child nodes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status

if TYPE_CHECKING:
    pass


class StackManagerHandler:
    """Handler for stack.manager_loop (house) nodes.

    Executes child nodes in sequence until a completion condition is met.
    The node's prompt specifies the loop body node IDs (comma-separated).
    Checks context for a 'stack_done' flag to exit the loop.

    Requires a registry to resolve child node handlers.
    """

    MAX_ITERATIONS = 100  # Safety limit to prevent infinite loops

    def __init__(self, registry: object | None = None) -> None:
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

        merged_updates: dict = {}
        iterations = 0

        while iterations < self.MAX_ITERATIONS:
            iterations += 1

            # Check for done flag before each iteration
            if context.get("stack_done"):
                break

            for child in children:
                if self._registry is not None:
                    handler = self._registry.resolve(child)  # type: ignore[attr-defined]
                    child_dir = logs_root / f"{child.id}_iter{iterations}"
                    child_dir.mkdir(parents=True, exist_ok=True)
                    outcome = handler.execute(child, context, graph, child_dir)

                    if outcome.context_updates:
                        context.apply_updates(outcome.context_updates)
                        merged_updates.update(outcome.context_updates)

                    if outcome.failed:
                        return Outcome(
                            status=Status.FAIL,
                            failure_reason=f"Child {child.id} failed on iteration {iterations}: {outcome.failure_reason}",
                            context_updates=merged_updates,
                        )

                # Check done flag after each child
                if context.get("stack_done"):
                    break

        merged_updates[f"{node.id}.iterations"] = iterations

        return Outcome(
            status=Status.SUCCESS,
            context_updates=merged_updates,
            notes=f"Loop completed after {iterations} iteration(s)",
        )
