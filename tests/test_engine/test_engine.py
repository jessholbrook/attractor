"""Tests for the execution engine core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from attractor.engine.engine import Engine, HandlerRegistry
from attractor.events.bus import EventBus
from attractor.events import types as events
from attractor.model.context import Context
from attractor.model.graph import Edge, Graph, Node
from attractor.model.outcome import Outcome, Status


# ---------------------------------------------------------------------------
# Stub handler
# ---------------------------------------------------------------------------

class StubHandler:
    """Test handler that returns a fixed outcome or a per-node outcome."""

    def __init__(self, outcome: Outcome | None = None) -> None:
        self.outcome = outcome or Outcome(status=Status.SUCCESS)
        self.calls: list[str] = []
        self._per_node: dict[str, list[Outcome]] = {}

    def set_outcomes(self, node_id: str, outcomes: list[Outcome]) -> None:
        """Set a sequence of outcomes for a specific node (for retry testing)."""
        self._per_node[node_id] = list(outcomes)

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        self.calls.append(node.id)
        if node.id in self._per_node and self._per_node[node.id]:
            return self._per_node[node.id].pop(0)
        return self.outcome


class ContextUpdatingHandler:
    """Handler that sets context keys based on node id."""

    def __init__(self, updates: dict[str, dict[str, Any]]) -> None:
        self._updates = updates
        self.calls: list[str] = []

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        self.calls.append(node.id)
        ctx_updates = self._updates.get(node.id, {})
        return Outcome(status=Status.SUCCESS, context_updates=ctx_updates)


class FailingHandler:
    """Handler that raises an exception."""

    def __init__(self, error_msg: str = "handler error") -> None:
        self.error_msg = error_msg
        self.calls: list[str] = []

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome:
        self.calls.append(node.id)
        raise RuntimeError(self.error_msg)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    name: str = "test_pipeline",
    nodes: list[Node] | None = None,
    edges: list[Edge] | None = None,
    attributes: dict[str, str] | None = None,
) -> Graph:
    """Construct a Graph from lists of nodes and edges."""
    g = Graph(name=name, attributes=attributes or {})
    for n in (nodes or []):
        g.nodes[n.id] = n
    g.edges = list(edges or [])
    return g


def make_linear_pipeline(
    node_ids: list[str],
    start_shape: str = "Mdiamond",
    exit_shape: str = "Msquare",
) -> Graph:
    """Build a linear pipeline: start -> A -> B -> ... -> exit."""
    nodes = []
    for i, nid in enumerate(node_ids):
        if i == 0:
            nodes.append(Node(id=nid, shape=start_shape))
        elif i == len(node_ids) - 1:
            nodes.append(Node(id=nid, shape=exit_shape))
        else:
            nodes.append(Node(id=nid, shape="box"))

    edges = []
    for i in range(len(node_ids) - 1):
        edges.append(Edge(from_node=node_ids[i], to_node=node_ids[i + 1]))

    return build_graph(nodes=nodes, edges=edges)


def make_registry(handler: Any) -> HandlerRegistry:
    """Create a HandlerRegistry with the given handler as default."""
    reg = HandlerRegistry()
    reg.set_default(handler)
    return reg


# ---------------------------------------------------------------------------
# Event collector
# ---------------------------------------------------------------------------

class EventCollector:
    """Collects all events emitted during a pipeline run."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[Any] = []
        bus.on_all(self.events.append)

    def of_type(self, event_type: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, event_type)]


# ---------------------------------------------------------------------------
# Tests: linear pipeline
# ---------------------------------------------------------------------------

class TestLinearPipeline:
    def test_simple_pipeline_completes(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "B", "exit"])
        handler = StubHandler()
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.SUCCESS

    def test_all_non_terminal_nodes_executed(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "B", "exit"])
        handler = StubHandler()
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        engine.run()
        assert handler.calls == ["start", "A", "B"]

    def test_manifest_written(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "exit"])
        handler = StubHandler()
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        engine.run()
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["name"] == "test_pipeline"


# ---------------------------------------------------------------------------
# Tests: context updates
# ---------------------------------------------------------------------------

class TestContextUpdates:
    def test_context_updates_visible_to_next_node(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "B", "exit"])
        ctx = Context()
        handler = ContextUpdatingHandler(
            updates={"A": {"result_a": "42"}}
        )
        engine = Engine(graph, make_registry(handler), context=ctx, logs_root=tmp_path)
        engine.run()
        assert ctx.get("result_a") == "42"

    def test_multiple_context_updates_accumulated(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "B", "exit"])
        ctx = Context()
        handler = ContextUpdatingHandler(
            updates={"start": {"x": "1"}, "A": {"y": "2"}, "B": {"z": "3"}}
        )
        engine = Engine(graph, make_registry(handler), context=ctx, logs_root=tmp_path)
        engine.run()
        assert ctx.get("x") == "1"
        assert ctx.get("y") == "2"
        assert ctx.get("z") == "3"


# ---------------------------------------------------------------------------
# Tests: goal gates
# ---------------------------------------------------------------------------

class TestGoalGates:
    def test_goal_gate_satisfied_allows_exit(self, tmp_path: Path):
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box", goal_gate=True),
            Node(id="exit", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="exit"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)
        handler = StubHandler(Outcome(status=Status.SUCCESS))
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.SUCCESS

    def test_goal_gate_unsatisfied_blocks_exit(self, tmp_path: Path):
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box", goal_gate=True),
            Node(id="exit", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="exit"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)
        handler = StubHandler(Outcome(status=Status.FAIL))
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.FAIL
        assert "Goal gate unsatisfied" in outcome.failure_reason

    def test_goal_gate_routes_to_retry_target(self, tmp_path: Path):
        """When a goal gate fails but a retry_target is set, route back to it."""
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box", goal_gate=True, retry_target="A"),
            Node(id="exit", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="exit"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        # Handler returns FAIL first, then SUCCESS on second call to A
        handler = StubHandler()
        handler.set_outcomes("A", [
            Outcome(status=Status.FAIL),
            Outcome(status=Status.SUCCESS),
        ])
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.SUCCESS
        # A was called twice (once fail, then routed back, then success)
        assert handler.calls.count("A") == 2


# ---------------------------------------------------------------------------
# Tests: checkpoint saving
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_checkpoint_saved_after_each_node(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "exit"])
        handler = StubHandler()
        bus = EventBus()
        collector = EventCollector(bus)
        engine = Engine(graph, make_registry(handler), event_bus=bus, logs_root=tmp_path)
        engine.run()
        cp_events = collector.of_type(events.CheckpointSaved)
        # One checkpoint per non-terminal node
        assert len(cp_events) == 2  # start, A
        # Checkpoint file exists
        assert (tmp_path / "checkpoint.json").exists()


# ---------------------------------------------------------------------------
# Tests: edge selection / conditions
# ---------------------------------------------------------------------------

class TestEdgeSelection:
    def test_condition_edge_selected_on_fail(self, tmp_path: Path):
        """When handler returns FAIL, a condition edge matching outcome=fail is followed."""
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box"),
            Node(id="ok", shape="Msquare"),
            Node(id="err", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="ok", condition="outcome=success"),
            Edge(from_node="A", to_node="err", condition="outcome=fail"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        handler = StubHandler()
        handler.set_outcomes("A", [Outcome(status=Status.FAIL)])
        # start succeeds, A fails
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        # Should have reached the "err" exit node
        assert outcome.status == Status.FAIL  # last_outcome from A

    def test_condition_edge_selected_on_success(self, tmp_path: Path):
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box"),
            Node(id="ok", shape="Msquare"),
            Node(id="err", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="ok", condition="outcome=success"),
            Edge(from_node="A", to_node="err", condition="outcome=fail"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        handler = StubHandler(Outcome(status=Status.SUCCESS))
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.SUCCESS


# ---------------------------------------------------------------------------
# Tests: retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retry_status_triggers_retry(self, tmp_path: Path):
        """A node returning RETRY should be retried up to max_retries."""
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box", max_retries=2),
            Node(id="exit", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="exit"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        handler = StubHandler()
        handler.set_outcomes("A", [
            Outcome(status=Status.RETRY),
            Outcome(status=Status.RETRY),
            Outcome(status=Status.SUCCESS),
        ])
        bus = EventBus()
        collector = EventCollector(bus)
        engine = Engine(graph, make_registry(handler), event_bus=bus, logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.SUCCESS
        # A called 3 times: 2 retries + 1 success
        assert handler.calls.count("A") == 3
        # Two retry events emitted
        retry_events = collector.of_type(events.StageRetrying)
        assert len(retry_events) == 2

    def test_retry_exhausted_returns_fail(self, tmp_path: Path):
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box", max_retries=1),
            Node(id="exit", shape="Msquare"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
            Edge(from_node="A", to_node="exit"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        handler = StubHandler()
        handler.set_outcomes("A", [
            Outcome(status=Status.RETRY),
            Outcome(status=Status.RETRY),
        ])
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        # With max_retries=1, max_attempts=2. Both return RETRY, so fail.
        assert outcome.status == Status.FAIL


# ---------------------------------------------------------------------------
# Tests: fail with no outgoing edge
# ---------------------------------------------------------------------------

class TestFailNoEdge:
    def test_fail_outcome_no_fail_edge_terminates(self, tmp_path: Path):
        """A node that fails with no outgoing edges causes pipeline termination."""
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box"),
            # No exit node reachable from A
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        handler = StubHandler(Outcome(status=Status.FAIL, failure_reason="broken"))
        engine = Engine(graph, make_registry(handler), logs_root=tmp_path)
        outcome = engine.run()
        assert outcome.status == Status.FAIL


# ---------------------------------------------------------------------------
# Tests: events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_pipeline_lifecycle_events(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "exit"])
        handler = StubHandler()
        bus = EventBus()
        collector = EventCollector(bus)
        engine = Engine(graph, make_registry(handler), event_bus=bus, logs_root=tmp_path)
        engine.run()

        # PipelineStarted
        started = collector.of_type(events.PipelineStarted)
        assert len(started) == 1
        assert started[0].graph_name == "test_pipeline"

        # StageStarted (start, A)
        stage_started = collector.of_type(events.StageStarted)
        assert len(stage_started) == 2
        assert stage_started[0].node_id == "start"
        assert stage_started[1].node_id == "A"

        # StageCompleted (start, A)
        stage_completed = collector.of_type(events.StageCompleted)
        assert len(stage_completed) == 2
        assert stage_completed[0].node_id == "start"
        assert stage_completed[1].node_id == "A"

        # PipelineCompleted
        completed = collector.of_type(events.PipelineCompleted)
        assert len(completed) == 1
        assert completed[0].graph_name == "test_pipeline"
        assert completed[0].outcome.status == Status.SUCCESS

    def test_pipeline_failed_event_on_failure(self, tmp_path: Path):
        nodes = [
            Node(id="start", shape="Mdiamond"),
            Node(id="A", shape="box"),
        ]
        edges = [
            Edge(from_node="start", to_node="A"),
        ]
        graph = build_graph(nodes=nodes, edges=edges)

        handler = StubHandler(Outcome(status=Status.FAIL, failure_reason="oops"))
        bus = EventBus()
        collector = EventCollector(bus)
        engine = Engine(graph, make_registry(handler), event_bus=bus, logs_root=tmp_path)
        engine.run()

        failed = collector.of_type(events.PipelineFailed)
        assert len(failed) == 1
        assert "A" in failed[0].error

    def test_checkpoint_saved_events(self, tmp_path: Path):
        graph = make_linear_pipeline(["start", "A", "B", "exit"])
        handler = StubHandler()
        bus = EventBus()
        collector = EventCollector(bus)
        engine = Engine(graph, make_registry(handler), event_bus=bus, logs_root=tmp_path)
        engine.run()

        cp_events = collector.of_type(events.CheckpointSaved)
        assert len(cp_events) == 3  # start, A, B
        node_ids = [e.node_id for e in cp_events]
        assert node_ids == ["start", "A", "B"]


# ---------------------------------------------------------------------------
# Tests: handler registry
# ---------------------------------------------------------------------------

class TestHandlerRegistry:
    def test_explicit_type_takes_priority(self):
        reg = HandlerRegistry()
        type_handler = StubHandler()
        default_handler = StubHandler()
        reg.register("custom", type_handler)
        reg.set_default(default_handler)
        node = Node(id="n", type="custom")
        assert reg.resolve(node) is type_handler

    def test_shape_based_resolution(self):
        reg = HandlerRegistry()
        start_handler = StubHandler()
        reg.register("start", start_handler)
        node = Node(id="n", shape="Mdiamond")
        assert reg.resolve(node) is start_handler

    def test_default_handler(self):
        reg = HandlerRegistry()
        default = StubHandler()
        reg.set_default(default)
        node = Node(id="n", shape="unknown_shape")
        assert reg.resolve(node) is default

    def test_no_handler_raises(self):
        reg = HandlerRegistry()
        node = Node(id="n", shape="unknown_shape")
        with pytest.raises(ValueError, match="No handler"):
            reg.resolve(node)
