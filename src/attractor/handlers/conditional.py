"""ConditionalHandler: evaluates condition expressions to choose a branch."""

from __future__ import annotations

from pathlib import Path

from attractor.conditions import evaluate_condition
from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class ConditionalHandler:
    """Handler for conditional (diamond) nodes.

    Evaluates the node's prompt as a condition expression against the context.
    Sets preferred_label to the first matching outgoing edge label.
    """

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        prompt = node.prompt or node.label
        outgoing = graph.outgoing_edges(node.id)

        # Evaluate each outgoing edge's condition/label to find a match.
        # The node's prompt is the condition expression to evaluate.
        # We check each outgoing edge: if its condition evaluates to true,
        # that edge's label becomes the preferred branch.
        for edge in outgoing:
            if edge.condition:
                # Build a temporary Outcome to evaluate the edge condition
                # against current context state.
                probe_outcome = Outcome(status=Status.SUCCESS)
                if evaluate_condition(edge.condition, probe_outcome, context):
                    return Outcome(
                        status=Status.SUCCESS,
                        preferred_label=edge.label,
                    )

        # If the prompt itself is a simple key=value expression,
        # evaluate it and pick the matching edge label.
        if prompt and "=" in prompt:
            probe_outcome = Outcome(status=Status.SUCCESS)
            for edge in outgoing:
                label_lower = edge.label.strip().lower()
                # Check if the condition result matches the edge label semantically
                result = evaluate_condition(prompt, probe_outcome, context)
                if result and label_lower in ("yes", "true", "y"):
                    return Outcome(
                        status=Status.SUCCESS,
                        preferred_label=edge.label,
                    )
                if not result and label_lower in ("no", "false", "n"):
                    return Outcome(
                        status=Status.SUCCESS,
                        preferred_label=edge.label,
                    )

        # Fallback: look up the prompt value in context and match against edge labels.
        if prompt:
            value = context.get(prompt, "")
            if value:
                value_str = str(value).strip().lower()
                for edge in outgoing:
                    if edge.label.strip().lower() == value_str:
                        return Outcome(
                            status=Status.SUCCESS,
                            preferred_label=edge.label,
                        )

        # No condition matched -- return SUCCESS with no preferred label;
        # the edge selector will use weight / lexical tiebreak.
        return Outcome(status=Status.SUCCESS)
