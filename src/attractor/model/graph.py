"""Core graph model: Node, Edge, and Graph dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node:
    """A single node in the pipeline graph."""

    id: str
    label: str = ""
    shape: str = "box"
    type: str = ""
    prompt: str = ""
    max_retries: int = 0
    goal_gate: bool = False
    retry_target: str = ""
    fallback_retry_target: str = ""
    fidelity: str = ""
    thread_id: str = ""
    node_class: str = ""
    timeout: float | None = None
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = "high"
    auto_status: bool = False
    allow_partial: bool = False
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Node id must be a non-empty string")

    @property
    def display_name(self) -> str:
        """Return the label if set, otherwise the id."""
        return self.label or self.id


@dataclass(frozen=True)
class Edge:
    """A directed edge connecting two nodes."""

    from_node: str
    to_node: str
    label: str = ""
    condition: str = ""
    weight: int = 0
    fidelity: str = ""
    thread_id: str = ""
    loop_restart: bool = False

    def __post_init__(self) -> None:
        if not self.from_node or not self.to_node:
            raise ValueError("Edge must have non-empty from_node and to_node")


@dataclass
class Graph:
    """The full pipeline graph containing nodes, edges, and metadata."""

    name: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def goal(self) -> str:
        """Return the pipeline-level goal string."""
        return self.attributes.get("goal", "")

    def start_node(self) -> Node | None:
        """Find the start node: first by Mdiamond shape, then by id."""
        for node in self.nodes.values():
            if node.shape == "Mdiamond":
                return node
        for name in ("start", "Start"):
            if name in self.nodes:
                return self.nodes[name]
        return None

    def exit_node(self) -> Node | None:
        """Find the exit node: first by Msquare shape, then by id."""
        for node in self.nodes.values():
            if node.shape == "Msquare":
                return node
        for name in ("exit", "end", "Exit", "End"):
            if name in self.nodes:
                return self.nodes[name]
        return None

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        """Return all edges originating from the given node."""
        return [e for e in self.edges if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        """Return all edges arriving at the given node."""
        return [e for e in self.edges if e.to_node == node_id]

    def reachable_from(self, node_id: str) -> set[str]:
        """Return all node IDs reachable from the given node via DFS."""
        visited: set[str] = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for edge in self.outgoing_edges(nid):
                stack.append(edge.to_node)
        return visited
