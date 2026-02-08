"""Validation rules for pipeline graphs.

Each rule is a function taking a Graph and returning a list of Diagnostic
objects describing any issues found.
"""

from __future__ import annotations

import re

from attractor.model.diagnostic import Diagnostic, Severity
from attractor.model.graph import Graph


# ---------------------------------------------------------------------------
# Known value sets
# ---------------------------------------------------------------------------

KNOWN_TYPES = frozenset({
    "start",
    "exit",
    "codergen",
    "wait.human",
    "conditional",
    "parallel",
    "parallel.fan_in",
    "tool",
    "stack.manager_loop",
})

VALID_FIDELITY = frozenset({
    "full",
    "truncate",
    "compact",
    "summary:low",
    "summary:medium",
    "summary:high",
})

# Shapes that map to the codergen handler (LLM nodes).
_LLM_SHAPES = frozenset({"box", "ellipse", "oval"})

# Condition clause pattern: key (=|!=) value
_CLAUSE_RE = re.compile(
    r"^\s*[A-Za-z_][A-Za-z0-9_.]*\s*(!?=)\s*[A-Za-z0-9_.:]*\s*$"
)


# ---------------------------------------------------------------------------
# Structural rules (ERROR severity)
# ---------------------------------------------------------------------------


def check_start_node(graph: Graph) -> list[Diagnostic]:
    """Exactly one start node (shape=Mdiamond) required."""
    starts = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    diagnostics: list[Diagnostic] = []
    if len(starts) == 0:
        diagnostics.append(
            Diagnostic(
                rule="check_start_node",
                severity=Severity.ERROR,
                message="No start node found. Exactly one node with shape=Mdiamond is required.",
                fix="Add a node with shape=Mdiamond to designate the start.",
            )
        )
    elif len(starts) > 1:
        for s in starts:
            diagnostics.append(
                Diagnostic(
                    rule="check_start_node",
                    severity=Severity.ERROR,
                    message=f"Multiple start nodes found. Node '{s.id}' has shape=Mdiamond.",
                    node_id=s.id,
                    fix="Ensure only one node has shape=Mdiamond.",
                )
            )
    return diagnostics


def check_terminal_node(graph: Graph) -> list[Diagnostic]:
    """At least one exit node (shape=Msquare) required."""
    exits = [n for n in graph.nodes.values() if n.shape == "Msquare"]
    if not exits:
        return [
            Diagnostic(
                rule="check_terminal_node",
                severity=Severity.ERROR,
                message="No exit node found. At least one node with shape=Msquare is required.",
                fix="Add a node with shape=Msquare to designate the exit.",
            )
        ]
    return []


def check_reachability(graph: Graph) -> list[Diagnostic]:
    """All nodes must be reachable from start node."""
    start = graph.start_node()
    if start is None:
        return []  # check_start_node will catch this
    reachable = graph.reachable_from(start.id)
    diagnostics: list[Diagnostic] = []
    for nid in graph.nodes:
        if nid not in reachable:
            diagnostics.append(
                Diagnostic(
                    rule="check_reachability",
                    severity=Severity.ERROR,
                    message=f"Node '{nid}' is not reachable from start node '{start.id}'.",
                    node_id=nid,
                    fix=f"Add an edge path from '{start.id}' to '{nid}'.",
                )
            )
    return diagnostics


def check_edge_target_exists(graph: Graph) -> list[Diagnostic]:
    """Every edge endpoint must reference an existing node."""
    diagnostics: list[Diagnostic] = []
    for edge in graph.edges:
        if edge.from_node not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="check_edge_target_exists",
                    severity=Severity.ERROR,
                    message=f"Edge references non-existent source node '{edge.from_node}'.",
                    edge=(edge.from_node, edge.to_node),
                    fix=f"Define node '{edge.from_node}' or fix the edge.",
                )
            )
        if edge.to_node not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="check_edge_target_exists",
                    severity=Severity.ERROR,
                    message=f"Edge references non-existent target node '{edge.to_node}'.",
                    edge=(edge.from_node, edge.to_node),
                    fix=f"Define node '{edge.to_node}' or fix the edge.",
                )
            )
    return diagnostics


def check_start_no_incoming(graph: Graph) -> list[Diagnostic]:
    """Start node must have no incoming edges."""
    start = graph.start_node()
    if start is None:
        return []
    incoming = graph.incoming_edges(start.id)
    if incoming:
        return [
            Diagnostic(
                rule="check_start_no_incoming",
                severity=Severity.ERROR,
                message=f"Start node '{start.id}' has {len(incoming)} incoming edge(s).",
                node_id=start.id,
                fix="Remove incoming edges to the start node.",
            )
        ]
    return []


def check_exit_no_outgoing(graph: Graph) -> list[Diagnostic]:
    """Exit nodes must have no outgoing edges."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.shape == "Msquare":
            outgoing = graph.outgoing_edges(node.id)
            if outgoing:
                diagnostics.append(
                    Diagnostic(
                        rule="check_exit_no_outgoing",
                        severity=Severity.ERROR,
                        message=f"Exit node '{node.id}' has {len(outgoing)} outgoing edge(s).",
                        node_id=node.id,
                        fix="Remove outgoing edges from the exit node.",
                    )
                )
    return diagnostics


def check_condition_syntax(graph: Graph) -> list[Diagnostic]:
    """Edge condition expressions must be syntactically valid.

    Valid syntax: ``key=value``, ``key!=value``, joined by ``&&``.
    """
    diagnostics: list[Diagnostic] = []
    for edge in graph.edges:
        cond = edge.condition.strip()
        if not cond:
            continue
        clauses = cond.split("&&")
        for clause in clauses:
            if not _CLAUSE_RE.match(clause):
                diagnostics.append(
                    Diagnostic(
                        rule="check_condition_syntax",
                        severity=Severity.ERROR,
                        message=f"Invalid condition syntax: '{clause.strip()}' in '{cond}'.",
                        edge=(edge.from_node, edge.to_node),
                        fix="Conditions must use key=value or key!=value separated by &&.",
                    )
                )
                break  # one diagnostic per edge is sufficient
    return diagnostics


# ---------------------------------------------------------------------------
# Semantic rules (WARNING severity)
# ---------------------------------------------------------------------------


def check_type_known(graph: Graph) -> list[Diagnostic]:
    """Node type values should be recognized. WARNING severity."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.type and node.type not in KNOWN_TYPES:
            diagnostics.append(
                Diagnostic(
                    rule="check_type_known",
                    severity=Severity.WARNING,
                    message=f"Node '{node.id}' has unrecognized type '{node.type}'.",
                    node_id=node.id,
                    fix=f"Use one of: {', '.join(sorted(KNOWN_TYPES))}.",
                )
            )
    return diagnostics


def check_fidelity_valid(graph: Graph) -> list[Diagnostic]:
    """Fidelity values must be valid. WARNING severity."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.fidelity and node.fidelity not in VALID_FIDELITY:
            diagnostics.append(
                Diagnostic(
                    rule="check_fidelity_valid",
                    severity=Severity.WARNING,
                    message=f"Node '{node.id}' has invalid fidelity '{node.fidelity}'.",
                    node_id=node.id,
                    fix=f"Use one of: {', '.join(sorted(VALID_FIDELITY))}.",
                )
            )
    return diagnostics


def check_retry_target_exists(graph: Graph) -> list[Diagnostic]:
    """retry_target / fallback_retry_target must reference existing nodes. WARNING."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.retry_target and node.retry_target not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="check_retry_target_exists",
                    severity=Severity.WARNING,
                    message=(
                        f"Node '{node.id}' has retry_target='{node.retry_target}' "
                        "which does not reference an existing node."
                    ),
                    node_id=node.id,
                    fix=f"Ensure node '{node.retry_target}' exists.",
                )
            )
        if node.fallback_retry_target and node.fallback_retry_target not in graph.nodes:
            diagnostics.append(
                Diagnostic(
                    rule="check_retry_target_exists",
                    severity=Severity.WARNING,
                    message=(
                        f"Node '{node.id}' has fallback_retry_target='{node.fallback_retry_target}' "
                        "which does not reference an existing node."
                    ),
                    node_id=node.id,
                    fix=f"Ensure node '{node.fallback_retry_target}' exists.",
                )
            )
    return diagnostics


def check_goal_gate_has_retry(graph: Graph) -> list[Diagnostic]:
    """goal_gate=true nodes should have a retry path. WARNING."""
    diagnostics: list[Diagnostic] = []
    # A graph-level retry_target counts for all goal_gate nodes.
    graph_retry = graph.attributes.get("retry_target", "")
    for node in graph.nodes.values():
        if node.goal_gate:
            has_retry = bool(node.retry_target or node.fallback_retry_target or graph_retry)
            if not has_retry:
                diagnostics.append(
                    Diagnostic(
                        rule="check_goal_gate_has_retry",
                        severity=Severity.WARNING,
                        message=(
                            f"Node '{node.id}' has goal_gate=true but no retry_target, "
                            "fallback_retry_target, or graph-level retry_target."
                        ),
                        node_id=node.id,
                        fix="Set retry_target on the node or at graph level.",
                    )
                )
    return diagnostics


def _is_llm_node(node_type: str, node_shape: str) -> bool:
    """Determine if a node resolves to the codergen (LLM) handler."""
    if node_type == "codergen":
        return True
    # Nodes without an explicit type that have a standard shape default to codergen.
    if not node_type and node_shape in _LLM_SHAPES:
        return True
    return False


def check_prompt_on_llm_nodes(graph: Graph) -> list[Diagnostic]:
    """Nodes resolving to codergen handler should have prompt or label. WARNING."""
    diagnostics: list[Diagnostic] = []
    for node in graph.nodes.values():
        # Skip start/exit nodes.
        if node.shape in ("Mdiamond", "Msquare"):
            continue
        if _is_llm_node(node.type, node.shape):
            if not node.prompt and not node.label:
                diagnostics.append(
                    Diagnostic(
                        rule="check_prompt_on_llm_nodes",
                        severity=Severity.WARNING,
                        message=f"LLM node '{node.id}' has neither prompt nor label.",
                        node_id=node.id,
                        fix="Add a prompt or label attribute to provide instructions.",
                    )
                )
    return diagnostics


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ALL_RULES = [
    check_start_node,
    check_terminal_node,
    check_reachability,
    check_edge_target_exists,
    check_start_no_incoming,
    check_exit_no_outgoing,
    check_condition_syntax,
    check_type_known,
    check_fidelity_valid,
    check_retry_target_exists,
    check_goal_gate_has_retry,
    check_prompt_on_llm_nodes,
]
