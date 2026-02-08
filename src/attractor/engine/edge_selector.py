"""Edge selection algorithm: 5-step priority for choosing the next edge."""

from __future__ import annotations

import re

from attractor.conditions import evaluate_condition
from attractor.model.context import Context
from attractor.model.graph import Edge
from attractor.model.outcome import Outcome


def select_edge(edges: list[Edge], outcome: Outcome, context: Context) -> Edge | None:
    """Select the next edge using 5-step priority:

    1. Condition match - edges with conditions that evaluate to true
    2. Preferred label - match outcome.preferred_label against edge labels (normalized)
    3. Suggested next IDs - match outcome.suggested_next_ids against edge targets
    4. Weight - highest weight among unconditional edges
    5. Lexical tiebreak - alphabetical by to_node
    """
    if not edges:
        return None

    # Step 1: Condition matching
    condition_matched = []
    for edge in edges:
        if edge.condition:
            if evaluate_condition(edge.condition, outcome, context):
                condition_matched.append(edge)
    if condition_matched:
        return _best_by_weight_then_lexical(condition_matched)

    # Step 2: Preferred label
    if outcome.preferred_label:
        for edge in edges:
            if _normalize_label(edge.label) == _normalize_label(outcome.preferred_label):
                return edge

    # Step 3: Suggested next IDs
    if outcome.suggested_next_ids:
        for sid in outcome.suggested_next_ids:
            for edge in edges:
                if edge.to_node == sid:
                    return edge

    # Step 4 & 5: Weight with lexical tiebreak (unconditional only)
    unconditional = [e for e in edges if not e.condition]
    if unconditional:
        return _best_by_weight_then_lexical(unconditional)

    # Fallback: any edge by weight then lexical
    return _best_by_weight_then_lexical(edges)


def _best_by_weight_then_lexical(edges: list[Edge]) -> Edge:
    """Return the edge with the highest weight, breaking ties alphabetically by to_node."""
    return sorted(edges, key=lambda e: (-e.weight, e.to_node))[0]


def _normalize_label(label: str) -> str:
    """Normalize: lowercase, strip whitespace, strip accelerator prefixes.

    Strips patterns like [K], K), K - from the beginning of labels.
    """
    label = label.strip().lower()
    # Strip [K] prefix, K) prefix, K - prefix
    label = re.sub(r"^\[[a-z]\]\s*", "", label)
    label = re.sub(r"^[a-z]\)\s*", "", label)
    label = re.sub(r"^[a-z]\s*-\s*", "", label)
    return label
