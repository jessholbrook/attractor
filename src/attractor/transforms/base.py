"""Base protocol for graph transforms."""

from __future__ import annotations

from typing import Protocol

from attractor.model.graph import Graph


class Transform(Protocol):
    """A graph-to-graph transformation step."""

    def apply(self, graph: Graph) -> Graph: ...
