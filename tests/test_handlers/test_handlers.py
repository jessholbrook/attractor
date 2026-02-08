"""Tests for all node handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from attractor.model.context import Context
from attractor.model.graph import Edge, Graph, Node
from attractor.model.outcome import Outcome, Status
from attractor.model.question import Answer, AnswerValue, Option, Question, QuestionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    nodes: list[Node] | None = None,
    edges: list[Edge] | None = None,
    name: str = "test-graph",
) -> Graph:
    node_dict = {n.id: n for n in (nodes or [])}
    return Graph(name=name, nodes=node_dict, edges=edges or [])


def _tmp_logs(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ===========================================================================
# StartHandler
# ===========================================================================


class TestStartHandler:
    def test_returns_success(self, tmp_path: Path) -> None:
        from attractor.handlers.start import StartHandler

        handler = StartHandler()
        node = Node(id="start", shape="Mdiamond")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS

    def test_sets_started_at(self, tmp_path: Path) -> None:
        from attractor.handlers.start import StartHandler

        handler = StartHandler()
        node = Node(id="start", shape="Mdiamond")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert "started_at" in outcome.context_updates
        assert isinstance(outcome.context_updates["started_at"], str)
        # Should be an ISO timestamp
        assert "T" in outcome.context_updates["started_at"]

    def test_started_at_is_utc_iso(self, tmp_path: Path) -> None:
        from datetime import datetime

        from attractor.handlers.start import StartHandler

        handler = StartHandler()
        node = Node(id="start", shape="Mdiamond")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        ts = outcome.context_updates["started_at"]
        # Should be parseable as an ISO datetime
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None


# ===========================================================================
# ExitHandler
# ===========================================================================


class TestExitHandler:
    def test_returns_success(self, tmp_path: Path) -> None:
        from attractor.handlers.exit_handler import ExitHandler

        handler = ExitHandler()
        node = Node(id="exit", shape="Msquare")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS

    def test_no_context_updates(self, tmp_path: Path) -> None:
        from attractor.handlers.exit_handler import ExitHandler

        handler = ExitHandler()
        node = Node(id="exit", shape="Msquare")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.context_updates == {}


# ===========================================================================
# ConditionalHandler
# ===========================================================================


class TestConditionalHandler:
    def test_matches_edge_condition(self, tmp_path: Path) -> None:
        from attractor.handlers.conditional import ConditionalHandler

        handler = ConditionalHandler()
        node = Node(id="cond1", shape="diamond", prompt="branch")
        edges = [
            Edge(from_node="cond1", to_node="a", label="yes", condition="outcome=success"),
            Edge(from_node="cond1", to_node="b", label="no", condition="outcome=fail"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        # outcome=success should match first edge
        assert outcome.status is Status.SUCCESS
        assert outcome.preferred_label == "yes"

    def test_no_match_returns_success_no_label(self, tmp_path: Path) -> None:
        from attractor.handlers.conditional import ConditionalHandler

        handler = ConditionalHandler()
        node = Node(id="cond1", shape="diamond", prompt="nonexistent_key")
        edges = [
            Edge(from_node="cond1", to_node="a", label="yes", condition="outcome=fail"),
            Edge(from_node="cond1", to_node="b", label="no", condition="outcome=retry"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS

    def test_context_lookup_matches_edge(self, tmp_path: Path) -> None:
        from attractor.handlers.conditional import ConditionalHandler

        handler = ConditionalHandler()
        node = Node(id="cond1", shape="diamond", prompt="direction")
        edges = [
            Edge(from_node="cond1", to_node="a", label="left"),
            Edge(from_node="cond1", to_node="b", label="right"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context({"direction": "right"})

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.preferred_label == "right"

    def test_condition_with_context_value(self, tmp_path: Path) -> None:
        from attractor.handlers.conditional import ConditionalHandler

        handler = ConditionalHandler()
        node = Node(id="cond1", shape="diamond", prompt="check")
        edges = [
            Edge(from_node="cond1", to_node="a", label="go", condition="ready=true"),
            Edge(from_node="cond1", to_node="b", label="wait", condition="ready=false"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context({"ready": "true"})

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.preferred_label == "go"


# ===========================================================================
# CodergenHandler
# ===========================================================================


class TestCodergenHandler:
    def test_calls_backend_returns_response(self, tmp_path: Path) -> None:
        from attractor.handlers.codergen import CodergenHandler, StubBackend

        backend = StubBackend()
        handler = CodergenHandler(backend)
        node = Node(id="gen1", shape="box", prompt="Write a hello world program")
        ctx = Context({"language": "python"})
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert "gen1.response" in outcome.context_updates
        assert "stub response:" in outcome.context_updates["gen1.response"]

    def test_returns_fail_on_backend_error(self, tmp_path: Path) -> None:
        from attractor.handlers.codergen import CodergenHandler

        class FailingBackend:
            def generate(self, prompt, context, *, model="", fidelity="", reasoning_effort="high"):
                raise RuntimeError("LLM service unavailable")

        handler = CodergenHandler(FailingBackend())
        node = Node(id="gen1", shape="box", prompt="Generate code")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.FAIL
        assert "LLM service unavailable" in outcome.failure_reason

    def test_uses_label_when_no_prompt(self, tmp_path: Path) -> None:
        from attractor.handlers.codergen import CodergenHandler, StubBackend

        backend = StubBackend()
        handler = CodergenHandler(backend)
        node = Node(id="gen1", shape="box", label="Generate docs")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert "Generate docs" in outcome.context_updates["gen1.response"]

    def test_passes_model_and_fidelity(self, tmp_path: Path) -> None:
        from attractor.handlers.codergen import CodergenHandler

        received: dict = {}

        class CapturingBackend:
            def generate(self, prompt, context, *, model="", fidelity="", reasoning_effort="high"):
                received["model"] = model
                received["fidelity"] = fidelity
                received["reasoning_effort"] = reasoning_effort
                return "ok"

        handler = CodergenHandler(CapturingBackend())
        node = Node(
            id="gen1",
            shape="box",
            prompt="test",
            llm_model="gpt-4",
            fidelity="high",
            reasoning_effort="medium",
        )
        ctx = Context()
        graph = _make_graph([node])

        handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert received["model"] == "gpt-4"
        assert received["fidelity"] == "high"
        assert received["reasoning_effort"] == "medium"

    def test_stub_backend_truncates_prompt(self) -> None:
        from attractor.handlers.codergen import StubBackend

        backend = StubBackend()
        long_prompt = "x" * 200
        result = backend.generate(long_prompt, {})
        assert result == f"stub response: {'x' * 50}"


# ===========================================================================
# WaitHumanHandler
# ===========================================================================


class TestWaitHumanHandler:
    def test_asks_interviewer_returns_answer(self, tmp_path: Path) -> None:
        from attractor.handlers.wait_human import WaitHumanHandler
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        handler = WaitHumanHandler(interviewer)

        node = Node(id="ask1", shape="hexagon", prompt="Continue?")
        edges = [
            Edge(from_node="ask1", to_node="a", label="Yes"),
            Edge(from_node="ask1", to_node="b", label="No"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.preferred_label == "YES"

    def test_multiple_choice_returns_first(self, tmp_path: Path) -> None:
        from attractor.handlers.wait_human import WaitHumanHandler
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        handler = WaitHumanHandler(interviewer)

        node = Node(id="ask1", shape="hexagon", prompt="Choose a language")
        edges = [
            Edge(from_node="ask1", to_node="py", label="Python"),
            Edge(from_node="ask1", to_node="rs", label="Rust"),
            Edge(from_node="ask1", to_node="ts", label="TypeScript"),
        ]
        graph = _make_graph(
            [node, Node(id="py"), Node(id="rs"), Node(id="ts")], edges
        )
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.preferred_label == "Python"

    def test_freeform_when_no_edges(self, tmp_path: Path) -> None:
        from attractor.handlers.wait_human import WaitHumanHandler
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        interviewer = AutoApproveInterviewer()
        handler = WaitHumanHandler(interviewer)

        node = Node(id="ask1", shape="hexagon", prompt="What is your name?")
        graph = _make_graph([node])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.preferred_label == "approved"

    def test_stores_answer_in_context(self, tmp_path: Path) -> None:
        from attractor.handlers.wait_human import WaitHumanHandler
        from attractor.interviewer.callback import CallbackInterviewer

        def always_go(q: Question) -> Answer:
            return Answer(text="go", value="go")

        handler = WaitHumanHandler(CallbackInterviewer(always_go))
        node = Node(id="ask1", shape="hexagon", prompt="Action?")
        graph = _make_graph([node])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.context_updates.get("ask1.answer") == "go"


# ===========================================================================
# ToolHandler
# ===========================================================================


class TestToolHandler:
    def test_calls_right_tool_by_id(self, tmp_path: Path) -> None:
        from attractor.handlers.tool import ToolHandler

        def my_tool(ctx: dict) -> dict:
            return {"result": "computed"}

        handler = ToolHandler({"tool1": my_tool})
        node = Node(id="tool1", shape="parallelogram", label="My Tool")
        ctx = Context({"input": "data"})
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.context_updates["result"] == "computed"

    def test_calls_right_tool_by_label(self, tmp_path: Path) -> None:
        from attractor.handlers.tool import ToolHandler

        def my_tool(ctx: dict) -> str:
            return "label_result"

        handler = ToolHandler({"My Tool": my_tool})
        node = Node(id="tool1", shape="parallelogram", label="My Tool")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.context_updates["tool1.result"] == "label_result"

    def test_returns_fail_for_unknown_tool(self, tmp_path: Path) -> None:
        from attractor.handlers.tool import ToolHandler

        handler = ToolHandler({})
        node = Node(id="unknown", shape="parallelogram", label="Unknown")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.FAIL
        assert "No tool registered" in outcome.failure_reason

    def test_returns_fail_on_tool_exception(self, tmp_path: Path) -> None:
        from attractor.handlers.tool import ToolHandler

        def bad_tool(ctx: dict) -> None:
            raise ValueError("tool broke")

        handler = ToolHandler({"tool1": bad_tool})
        node = Node(id="tool1", shape="parallelogram")
        ctx = Context()
        graph = _make_graph([node])

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.FAIL
        assert "tool broke" in outcome.failure_reason

    def test_tool_receives_context_snapshot(self, tmp_path: Path) -> None:
        from attractor.handlers.tool import ToolHandler

        received_ctx: dict = {}

        def capture_tool(ctx: dict) -> None:
            received_ctx.update(ctx)

        handler = ToolHandler({"tool1": capture_tool})
        node = Node(id="tool1", shape="parallelogram")
        ctx = Context({"key1": "val1", "key2": 42})
        graph = _make_graph([node])

        handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert received_ctx["key1"] == "val1"
        assert received_ctx["key2"] == 42


# ===========================================================================
# ParallelHandler
# ===========================================================================


class TestParallelHandler:
    def test_runs_children_concurrently(self, tmp_path: Path) -> None:
        from attractor.handlers.parallel import ParallelHandler

        handler = ParallelHandler(registry=None)
        child_a = Node(id="child_a", shape="box")
        child_b = Node(id="child_b", shape="box")
        node = Node(id="par1", shape="component", prompt="child_a, child_b")
        graph = _make_graph([node, child_a, child_b])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert "2/2" in outcome.notes

    def test_returns_fail_when_all_fail(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.parallel import ParallelHandler

        class FailHandler:
            def execute(self, node, context, graph, logs_root):
                return Outcome(status=Status.FAIL, failure_reason="always fail")

        reg = HandlerRegistry()
        reg.set_default(FailHandler())
        handler = ParallelHandler(registry=reg)

        child_a = Node(id="child_a", shape="box")
        child_b = Node(id="child_b", shape="box")
        node = Node(id="par1", shape="component", prompt="child_a, child_b")
        graph = _make_graph([node, child_a, child_b])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.FAIL

    def test_returns_partial_on_mixed(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.parallel import ParallelHandler

        call_count = 0

        class AlternatingHandler:
            def execute(self, node, context, graph, logs_root):
                nonlocal call_count
                call_count += 1
                if node.id == "child_a":
                    return Outcome(status=Status.SUCCESS)
                return Outcome(status=Status.FAIL, failure_reason="partial fail")

        reg = HandlerRegistry()
        reg.set_default(AlternatingHandler())
        handler = ParallelHandler(registry=reg)

        child_a = Node(id="child_a", shape="box")
        child_b = Node(id="child_b", shape="box")
        node = Node(id="par1", shape="component", prompt="child_a, child_b")
        graph = _make_graph([node, child_a, child_b])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.PARTIAL_SUCCESS

    def test_merges_context_updates(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.parallel import ParallelHandler

        class UpdateHandler:
            def execute(self, node, context, graph, logs_root):
                return Outcome(
                    status=Status.SUCCESS,
                    context_updates={f"{node.id}.done": True},
                )

        reg = HandlerRegistry()
        reg.set_default(UpdateHandler())
        handler = ParallelHandler(registry=reg)

        child_a = Node(id="child_a", shape="box")
        child_b = Node(id="child_b", shape="box")
        node = Node(id="par1", shape="component", prompt="child_a, child_b")
        graph = _make_graph([node, child_a, child_b])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.context_updates.get("child_a.done") is True
        assert outcome.context_updates.get("child_b.done") is True

    def test_no_children_returns_success(self, tmp_path: Path) -> None:
        from attractor.handlers.parallel import ParallelHandler

        handler = ParallelHandler()
        node = Node(id="par1", shape="component", prompt="")
        graph = _make_graph([node])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS

    def test_invalid_child_ids_returns_fail(self, tmp_path: Path) -> None:
        from attractor.handlers.parallel import ParallelHandler

        handler = ParallelHandler()
        node = Node(id="par1", shape="component", prompt="nonexistent_a, nonexistent_b")
        graph = _make_graph([node])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.FAIL


# ===========================================================================
# FanInHandler
# ===========================================================================


class TestFanInHandler:
    def test_all_predecessors_done(self, tmp_path: Path) -> None:
        from attractor.handlers.fan_in import FanInHandler

        handler = FanInHandler()
        node = Node(id="fan_in", shape="tripleoctagon")
        edges = [
            Edge(from_node="a", to_node="fan_in"),
            Edge(from_node="b", to_node="fan_in"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context({"a.complete": True, "b.complete": True})

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS

    def test_missing_predecessor_returns_retry(self, tmp_path: Path) -> None:
        from attractor.handlers.fan_in import FanInHandler

        handler = FanInHandler()
        node = Node(id="fan_in", shape="tripleoctagon")
        edges = [
            Edge(from_node="a", to_node="fan_in"),
            Edge(from_node="b", to_node="fan_in"),
        ]
        graph = _make_graph([node, Node(id="a"), Node(id="b")], edges)
        ctx = Context({"a.complete": True})  # b is not complete

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.RETRY
        assert "b" in outcome.notes

    def test_no_predecessors_returns_success(self, tmp_path: Path) -> None:
        from attractor.handlers.fan_in import FanInHandler

        handler = FanInHandler()
        node = Node(id="fan_in", shape="tripleoctagon")
        graph = _make_graph([node])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS


# ===========================================================================
# StackManagerHandler
# ===========================================================================


class TestStackManagerHandler:
    def test_loops_until_done_flag(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.stack_manager import StackManagerHandler

        iteration_count = 0

        class CountHandler:
            def execute(self, node, context, graph, logs_root):
                nonlocal iteration_count
                iteration_count += 1
                if iteration_count >= 3:
                    context.set("stack_done", True)
                return Outcome(status=Status.SUCCESS)

        reg = HandlerRegistry()
        reg.set_default(CountHandler())
        handler = StackManagerHandler(registry=reg)

        child = Node(id="child1", shape="box")
        node = Node(id="loop1", shape="house", prompt="child1")
        graph = _make_graph([node, child])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert iteration_count == 3

    def test_no_children_returns_success(self, tmp_path: Path) -> None:
        from attractor.handlers.stack_manager import StackManagerHandler

        handler = StackManagerHandler()
        node = Node(id="loop1", shape="house", prompt="")
        graph = _make_graph([node])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS

    def test_child_failure_stops_loop(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.stack_manager import StackManagerHandler

        class FailHandler:
            def execute(self, node, context, graph, logs_root):
                return Outcome(status=Status.FAIL, failure_reason="child error")

        reg = HandlerRegistry()
        reg.set_default(FailHandler())
        handler = StackManagerHandler(registry=reg)

        child = Node(id="child1", shape="box")
        node = Node(id="loop1", shape="house", prompt="child1")
        graph = _make_graph([node, child])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.FAIL
        assert "child error" in outcome.failure_reason

    def test_already_done_exits_immediately(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.stack_manager import StackManagerHandler

        call_count = 0

        class TrackHandler:
            def execute(self, node, context, graph, logs_root):
                nonlocal call_count
                call_count += 1
                return Outcome(status=Status.SUCCESS)

        reg = HandlerRegistry()
        reg.set_default(TrackHandler())
        handler = StackManagerHandler(registry=reg)

        child = Node(id="child1", shape="box")
        node = Node(id="loop1", shape="house", prompt="child1")
        graph = _make_graph([node, child])
        ctx = Context({"stack_done": True})

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert call_count == 0

    def test_merges_context_updates(self, tmp_path: Path) -> None:
        from attractor.engine.engine import HandlerRegistry
        from attractor.handlers.stack_manager import StackManagerHandler

        class SetAndDone:
            def execute(self, node, context, graph, logs_root):
                context.set("stack_done", True)
                return Outcome(
                    status=Status.SUCCESS,
                    context_updates={"data": "value"},
                )

        reg = HandlerRegistry()
        reg.set_default(SetAndDone())
        handler = StackManagerHandler(registry=reg)

        child = Node(id="child1", shape="box")
        node = Node(id="loop1", shape="house", prompt="child1")
        graph = _make_graph([node, child])
        ctx = Context()

        outcome = handler.execute(node, ctx, graph, _tmp_logs(tmp_path))

        assert outcome.status is Status.SUCCESS
        assert outcome.context_updates.get("data") == "value"


# ===========================================================================
# create_default_registry
# ===========================================================================


class TestCreateDefaultRegistry:
    def test_creates_registry_with_all_types(self) -> None:
        from attractor.handlers import create_default_registry

        registry = create_default_registry()

        # Verify all handler types are registered by resolving nodes with matching shapes
        for shape, type_name in [
            ("Mdiamond", "start"),
            ("Msquare", "exit"),
            ("box", "codergen"),
            ("diamond", "conditional"),
            ("component", "parallel"),
            ("tripleoctagon", "parallel.fan_in"),
            ("parallelogram", "tool"),
            ("house", "stack.manager_loop"),
        ]:
            node = Node(id=f"test_{type_name}", shape=shape)
            handler = registry.resolve(node)
            assert handler is not None

    def test_with_custom_backend(self) -> None:
        from attractor.handlers import create_default_registry
        from attractor.handlers.codergen import StubBackend

        backend = StubBackend()
        registry = create_default_registry(codergen_backend=backend)

        node = Node(id="gen", shape="box")
        handler = registry.resolve(node)
        assert handler is not None

    def test_with_interviewer(self) -> None:
        from attractor.handlers import create_default_registry
        from attractor.interviewer.auto_approve import AutoApproveInterviewer

        registry = create_default_registry(interviewer=AutoApproveInterviewer())

        node = Node(id="ask", shape="hexagon")
        handler = registry.resolve(node)
        assert handler is not None
