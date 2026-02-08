from __future__ import annotations

from wolverine.pipeline.graph import build_wolverine_graph


class TestWolverineGraph:
    def test_node_count(self) -> None:
        graph = build_wolverine_graph()
        assert len(graph.nodes) == 12

    def test_edge_count(self) -> None:
        graph = build_wolverine_graph()
        assert len(graph.edges) == 15

    def test_start_node_found(self) -> None:
        graph = build_wolverine_graph()
        start = graph.start_node()
        assert start is not None
        assert start.id == "Start"
        assert start.shape == "Mdiamond"

    def test_exit_node_found(self) -> None:
        graph = build_wolverine_graph()
        exit_node = graph.exit_node()
        assert exit_node is not None
        assert exit_node.id == "Exit"
        assert exit_node.shape == "Msquare"

    def test_node_shapes(self) -> None:
        graph = build_wolverine_graph()
        expected_shapes = {
            "Start": "Mdiamond",
            "ingest": "parallelogram",
            "classify": "box",
            "is_duplicate": "diamond",
            "deduplicate": "parallelogram",
            "diagnose": "box",
            "heal": "box",
            "validate": "diamond",
            "review_gate": "hexagon",
            "apply": "parallelogram",
            "revise": "box",
            "Exit": "Msquare",
        }
        for node_id, expected_shape in expected_shapes.items():
            assert graph.nodes[node_id].shape == expected_shape, (
                f"Node {node_id}: expected shape {expected_shape}, "
                f"got {graph.nodes[node_id].shape}"
            )

    def test_conditions_on_edges(self) -> None:
        graph = build_wolverine_graph()
        conditional_edges = [e for e in graph.edges if e.condition]
        assert len(conditional_edges) == 4
        conditions = {(e.from_node, e.to_node): e.condition for e in conditional_edges}
        assert conditions[("is_duplicate", "deduplicate")] == "context.is_duplicate=true"
        assert conditions[("is_duplicate", "diagnose")] == "context.is_duplicate!=true"
        assert conditions[("validate", "review_gate")] == "outcome=success"
        assert conditions[("validate", "heal")] == "outcome=fail"

    def test_happy_path_exists(self) -> None:
        """Start -> ingest -> classify -> is_duplicate -> diagnose -> heal -> validate -> review_gate -> apply -> Exit."""
        graph = build_wolverine_graph()
        path = ["Start", "ingest", "classify", "is_duplicate", "diagnose", "heal", "validate", "review_gate", "apply", "Exit"]
        for i in range(len(path) - 1):
            edges = graph.outgoing_edges(path[i])
            targets = [e.to_node for e in edges]
            assert path[i + 1] in targets, (
                f"Expected edge from {path[i]} to {path[i+1]}, "
                f"but outgoing targets are {targets}"
            )

    def test_duplicate_path_exists(self) -> None:
        """is_duplicate -> deduplicate -> Exit."""
        graph = build_wolverine_graph()
        dedup_edges = graph.outgoing_edges("is_duplicate")
        assert any(e.to_node == "deduplicate" for e in dedup_edges)
        exit_edges = graph.outgoing_edges("deduplicate")
        assert any(e.to_node == "Exit" for e in exit_edges)

    def test_reject_path_exists(self) -> None:
        """review_gate -> Exit (Reject)."""
        graph = build_wolverine_graph()
        review_edges = graph.outgoing_edges("review_gate")
        reject_edge = [e for e in review_edges if e.label == "Reject"]
        assert len(reject_edge) == 1
        assert reject_edge[0].to_node == "Exit"

    def test_revision_path_exists(self) -> None:
        """review_gate -> revise -> validate."""
        graph = build_wolverine_graph()
        review_edges = graph.outgoing_edges("review_gate")
        changes_edge = [e for e in review_edges if e.label == "Request Changes"]
        assert len(changes_edge) == 1
        assert changes_edge[0].to_node == "revise"
        revise_edges = graph.outgoing_edges("revise")
        assert any(e.to_node == "validate" for e in revise_edges)

    def test_graph_name(self) -> None:
        graph = build_wolverine_graph()
        assert graph.name == "wolverine"

    def test_graph_goal(self) -> None:
        graph = build_wolverine_graph()
        assert graph.goal == "Self-heal a software issue from signal to reviewed solution"
