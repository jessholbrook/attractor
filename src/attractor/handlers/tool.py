"""ToolHandler: dispatches to registered tool functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class ToolHandler:
    """Handler for tool (parallelogram) nodes.

    Looks up a tool function by node ID or label in the tool registry,
    calls it with a context snapshot, and returns the result as context updates.
    """

    def __init__(self, tool_registry: dict[str, Callable[..., Any]]) -> None:
        self._registry = tool_registry

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        # Try to find the tool by node ID first, then by label
        tool_fn = self._registry.get(node.id) or self._registry.get(node.label)

        if tool_fn is None:
            return Outcome(
                status=Status.FAIL,
                failure_reason=f"No tool registered for node '{node.id}' or label '{node.label}'",
            )

        snapshot = context.snapshot()

        try:
            result = tool_fn(snapshot)
        except Exception as exc:
            return Outcome(
                status=Status.FAIL,
                failure_reason=f"Tool '{node.id}' raised: {exc}",
            )

        # The tool can return a dict (context updates), a string, or None
        if isinstance(result, dict):
            updates = result
        elif result is not None:
            updates = {f"{node.id}.result": result}
        else:
            updates = {}

        return Outcome(
            status=Status.SUCCESS,
            context_updates=updates,
        )
