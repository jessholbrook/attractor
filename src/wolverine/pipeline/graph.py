from __future__ import annotations

from attractor.model.graph import Edge, Graph, Node


def build_wolverine_graph() -> Graph:
    """Build the self-healing pipeline graph."""
    nodes = {
        "Start": Node(id="Start", label="Start", shape="Mdiamond"),
        "ingest": Node(id="ingest", label="Ingest Signal", shape="parallelogram", type="tool"),
        "classify": Node(
            id="classify",
            label="Classify & Triage",
            shape="box",
            prompt="Classify this signal by severity, category, and affected area. Check for duplicates.",
        ),
        "is_duplicate": Node(
            id="is_duplicate", label="Duplicate?", shape="diamond", type="conditional"
        ),
        "deduplicate": Node(
            id="deduplicate", label="Merge Duplicate", shape="parallelogram", type="tool"
        ),
        "diagnose": Node(
            id="diagnose",
            label="Diagnose Root Cause",
            shape="box",
            prompt="Analyze this issue to identify root cause, affected files, and fix strategy.",
        ),
        "heal": Node(
            id="heal",
            label="Generate Solution",
            shape="box",
            prompt="Generate a concrete code fix for the diagnosed issue.",
        ),
        "validate": Node(
            id="validate", label="Validate Solution", shape="diamond", type="conditional"
        ),
        "review_gate": Node(
            id="review_gate", label="Human Review", shape="hexagon", type="wait.human"
        ),
        "apply": Node(
            id="apply", label="Record Approval", shape="parallelogram", type="tool"
        ),
        "revise": Node(
            id="revise",
            label="Revise with Feedback",
            shape="box",
            prompt="Incorporate reviewer feedback and generate an updated solution.",
        ),
        "Exit": Node(id="Exit", label="Exit", shape="Msquare"),
    }

    edges = [
        Edge(from_node="Start", to_node="ingest"),
        Edge(from_node="ingest", to_node="classify"),
        Edge(from_node="classify", to_node="is_duplicate"),
        Edge(
            from_node="is_duplicate",
            to_node="deduplicate",
            label="Yes",
            condition="context.is_duplicate=true",
        ),
        Edge(
            from_node="is_duplicate",
            to_node="diagnose",
            label="No",
            condition="context.is_duplicate!=true",
        ),
        Edge(from_node="deduplicate", to_node="Exit", label="Merged"),
        Edge(from_node="diagnose", to_node="heal"),
        Edge(from_node="heal", to_node="validate"),
        Edge(
            from_node="validate",
            to_node="review_gate",
            label="Valid",
            condition="outcome=success",
        ),
        Edge(
            from_node="validate",
            to_node="heal",
            label="Invalid",
            condition="outcome=fail",
        ),
        Edge(from_node="review_gate", to_node="apply", label="Approve"),
        Edge(from_node="review_gate", to_node="revise", label="Request Changes"),
        Edge(from_node="review_gate", to_node="Exit", label="Reject"),
        Edge(from_node="revise", to_node="validate"),
        Edge(from_node="apply", to_node="Exit"),
    ]

    return Graph(
        name="wolverine",
        nodes=nodes,
        edges=edges,
        attributes={"goal": "Self-heal a software issue from signal to reviewed solution"},
    )
