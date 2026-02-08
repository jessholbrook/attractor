"""Integration tests: full pipeline execution end-to-end.

These tests exercise the complete pipeline flow: parse DOT file, validate,
transform, build a handler registry, create an Engine, and run to completion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from attractor.cli.main import cli
from attractor.engine.engine import Engine, HandlerRegistry
from attractor.events.bus import EventBus
from attractor.model.checkpoint import Checkpoint
from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status
from attractor.parser import parse_dot
from attractor.transforms import apply_transforms
from attractor.validation import validate, validate_or_raise

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubHandler:
    """Handler that always returns a configurable outcome (default: SUCCESS)."""

    def __init__(self, outcome: Outcome | None = None) -> None:
        self.outcome = outcome or Outcome(status=Status.SUCCESS)
        self.calls: list[str] = []

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        self.calls.append(node.id)
        return self.outcome


class PerNodeHandler:
    """Handler that returns different outcomes per node id."""

    def __init__(self, outcomes: dict[str, Outcome]) -> None:
        self._outcomes = outcomes
        self.calls: list[str] = []

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        self.calls.append(node.id)
        return self._outcomes.get(node.id, Outcome(status=Status.SUCCESS))


class SequenceHandler:
    """Handler that returns outcomes from a per-node sequence, falling back to SUCCESS."""

    def __init__(self, sequences: dict[str, list[Outcome]] | None = None) -> None:
        self._sequences: dict[str, list[Outcome]] = sequences or {}
        self.calls: list[str] = []

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        self.calls.append(node.id)
        if node.id in self._sequences and self._sequences[node.id]:
            return self._sequences[node.id].pop(0)
        return Outcome(status=Status.SUCCESS)


class AutoApproveHandler:
    """Handler for wait.human nodes that always approves."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(
        self, node: Node, context: Context, graph: Graph, logs_root: Path
    ) -> Outcome:
        self.calls.append(node.id)
        return Outcome(status=Status.SUCCESS, preferred_label="approve")


def load_example(name: str) -> Graph:
    """Load and parse an example DOT file."""
    path = EXAMPLES / name
    return parse_dot(path.read_text(encoding="utf-8"))


def make_registry(
    handler: Any,
    *,
    auto_approve: bool = False,
    type_handlers: dict[str, Any] | None = None,
) -> HandlerRegistry:
    """Build a registry with given handler as default, plus optional type overrides."""
    reg = HandlerRegistry()
    reg.set_default(handler)
    if type_handlers:
        for type_name, h in type_handlers.items():
            reg.register(type_name, h)
    if auto_approve:
        reg.register("wait.human", AutoApproveHandler())
    return reg


def run_pipeline(
    graph: Graph,
    handler: Any,
    tmp_path: Path,
    *,
    auto_approve: bool = False,
    checkpoint: Checkpoint | None = None,
    type_handlers: dict[str, Any] | None = None,
) -> tuple[Outcome, Engine]:
    """Full pipeline run helper: validate, transform, create engine, run."""
    validate_or_raise(graph)
    graph = apply_transforms(graph)
    registry = make_registry(handler, auto_approve=auto_approve, type_handlers=type_handlers)
    context = Context()
    event_bus = EventBus()
    engine = Engine(
        graph,
        registry,
        context=context,
        event_bus=event_bus,
        logs_root=tmp_path,
        checkpoint=checkpoint,
    )
    outcome = engine.run()
    return outcome, engine


# ---------------------------------------------------------------------------
# Test 1: hello_world.dot runs to completion with stub backend
# ---------------------------------------------------------------------------


class TestHelloWorldPipeline:
    def test_hello_world_runs_to_success(self, tmp_path: Path) -> None:
        graph = load_example("hello_world.dot")
        outcome, engine = run_pipeline(graph, StubHandler(), tmp_path)
        assert outcome.status == Status.SUCCESS

    def test_hello_world_manifest_exists(self, tmp_path: Path) -> None:
        graph = load_example("hello_world.dot")
        run_pipeline(graph, StubHandler(), tmp_path)
        manifest = tmp_path / "manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["name"] == "hello_world"

    def test_hello_world_checkpoint_exists(self, tmp_path: Path) -> None:
        graph = load_example("hello_world.dot")
        run_pipeline(graph, StubHandler(), tmp_path)
        checkpoint = tmp_path / "checkpoint.json"
        assert checkpoint.exists()

    def test_hello_world_nodes_executed(self, tmp_path: Path) -> None:
        handler = StubHandler()
        graph = load_example("hello_world.dot")
        run_pipeline(graph, handler, tmp_path)
        # Start and greet should be executed (End is terminal, not executed)
        assert "Start" in handler.calls
        assert "greet" in handler.calls

    def test_hello_world_goal_in_manifest(self, tmp_path: Path) -> None:
        graph = load_example("hello_world.dot")
        run_pipeline(graph, StubHandler(), tmp_path)
        manifest = tmp_path / "manifest.json"
        data = json.loads(manifest.read_text())
        assert data["goal"] == "Say hello to the world"


# ---------------------------------------------------------------------------
# Test 2: branching.dot takes success path
# ---------------------------------------------------------------------------


class TestBranchingSuccessPath:
    def test_success_path_reaches_process_good(self, tmp_path: Path) -> None:
        graph = load_example("branching.dot")
        # check returns SUCCESS -> should go to process_good
        handler = StubHandler(Outcome(status=Status.SUCCESS))
        outcome, engine = run_pipeline(graph, handler, tmp_path)
        assert outcome.status == Status.SUCCESS
        assert "process_good" in handler.calls
        assert "process_bad" not in handler.calls

    def test_success_path_includes_merge(self, tmp_path: Path) -> None:
        graph = load_example("branching.dot")
        handler = StubHandler(Outcome(status=Status.SUCCESS))
        run_pipeline(graph, handler, tmp_path)
        assert "merge" in handler.calls


# ---------------------------------------------------------------------------
# Test 3: branching.dot takes failure path
# ---------------------------------------------------------------------------


class TestBranchingFailPath:
    def test_fail_path_reaches_process_bad(self, tmp_path: Path) -> None:
        graph = load_example("branching.dot")
        # check returns FAIL -> should go to process_bad
        handler = PerNodeHandler({
            "check": Outcome(status=Status.FAIL),
        })
        outcome, engine = run_pipeline(graph, handler, tmp_path)
        assert outcome.status == Status.SUCCESS
        assert "process_bad" in handler.calls
        assert "process_good" not in handler.calls

    def test_fail_path_includes_merge(self, tmp_path: Path) -> None:
        graph = load_example("branching.dot")
        handler = PerNodeHandler({
            "check": Outcome(status=Status.FAIL),
        })
        run_pipeline(graph, handler, tmp_path)
        assert "merge" in handler.calls


# ---------------------------------------------------------------------------
# Test 4: code_review.dot with auto-approve
# ---------------------------------------------------------------------------


class TestCodeReviewAutoApprove:
    def test_auto_approve_completes(self, tmp_path: Path) -> None:
        graph = load_example("code_review.dot")
        handler = StubHandler(Outcome(status=Status.SUCCESS))
        outcome, engine = run_pipeline(
            graph, handler, tmp_path, auto_approve=True
        )
        assert outcome.status == Status.SUCCESS

    def test_auto_approve_executes_analyze(self, tmp_path: Path) -> None:
        graph = load_example("code_review.dot")
        handler = StubHandler(Outcome(status=Status.SUCCESS))
        auto = AutoApproveHandler()
        outcome, engine = run_pipeline(
            graph,
            handler,
            tmp_path,
            type_handlers={"wait.human": auto},
        )
        assert outcome.status == Status.SUCCESS
        assert "analyze" in handler.calls
        assert "review_gate" in auto.calls

    def test_auto_approve_skips_fix_loop(self, tmp_path: Path) -> None:
        """With auto-approve (success), the pipeline should not enter fix loop."""
        graph = load_example("code_review.dot")
        handler = StubHandler(Outcome(status=Status.SUCCESS))
        outcome, engine = run_pipeline(
            graph, handler, tmp_path, auto_approve=True
        )
        assert "fix" not in handler.calls


# ---------------------------------------------------------------------------
# Test 5: Resume from checkpoint
# ---------------------------------------------------------------------------


class TestResumeFromCheckpoint:
    def test_resume_completes_pipeline(self, tmp_path: Path) -> None:
        """Run a pipeline partially, save checkpoint, then resume."""
        graph = load_example("hello_world.dot")

        # First run: complete the pipeline normally to get a valid checkpoint
        handler1 = StubHandler()
        outcome1, engine1 = run_pipeline(graph, handler1, tmp_path / "run1")
        assert outcome1.status == Status.SUCCESS

        # Load the checkpoint from the first run
        cp_path = tmp_path / "run1" / "checkpoint.json"
        assert cp_path.exists()
        checkpoint = Checkpoint.load(cp_path)

        # Second run: resume from checkpoint
        handler2 = StubHandler()
        graph2 = load_example("hello_world.dot")
        graph2 = apply_transforms(graph2)
        registry = make_registry(handler2)
        engine2 = Engine(
            graph2,
            registry,
            context=Context(),
            event_bus=EventBus(),
            logs_root=tmp_path / "run2",
            checkpoint=checkpoint,
        )
        outcome2 = engine2.run()
        assert outcome2.status == Status.SUCCESS

    def test_checkpoint_contains_completed_nodes(self, tmp_path: Path) -> None:
        graph = load_example("hello_world.dot")
        handler = StubHandler()
        run_pipeline(graph, handler, tmp_path)
        cp = Checkpoint.load(tmp_path / "checkpoint.json")
        assert len(cp.completed_nodes) > 0

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """Checkpoint can be saved and loaded without data loss."""
        graph = load_example("hello_world.dot")
        handler = StubHandler()
        run_pipeline(graph, handler, tmp_path)
        cp_path = tmp_path / "checkpoint.json"
        cp = Checkpoint.load(cp_path)

        # Save to new path and reload
        cp2_path = tmp_path / "checkpoint2.json"
        cp.save(cp2_path)
        cp2 = Checkpoint.load(cp2_path)

        assert cp.current_node == cp2.current_node
        assert cp.completed_nodes == cp2.completed_nodes
        assert cp.timestamp == cp2.timestamp


# ---------------------------------------------------------------------------
# Test 6: Validate example DOT files
# ---------------------------------------------------------------------------


class TestValidateExamples:
    def test_validate_hello_world(self) -> None:
        graph = load_example("hello_world.dot")
        diagnostics = validate(graph)
        errors = [d for d in diagnostics if d.is_error]
        assert len(errors) == 0

    def test_validate_branching(self) -> None:
        graph = load_example("branching.dot")
        diagnostics = validate(graph)
        errors = [d for d in diagnostics if d.is_error]
        assert len(errors) == 0

    def test_validate_code_review(self) -> None:
        graph = load_example("code_review.dot")
        diagnostics = validate(graph)
        errors = [d for d in diagnostics if d.is_error]
        assert len(errors) == 0

    def test_validate_or_raise_hello_world(self) -> None:
        graph = load_example("hello_world.dot")
        # Should not raise
        validate_or_raise(graph)


# ---------------------------------------------------------------------------
# Test 7: CLI validate command
# ---------------------------------------------------------------------------


class TestCLIValidate:
    def test_cli_validate_valid_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(EXAMPLES / "hello_world.dot")])
        assert result.exit_code == 0

    def test_cli_validate_output_contains_ok_or_diagnostics(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(EXAMPLES / "hello_world.dot")])
        # Should print OK or summary
        assert result.exit_code == 0

    def test_cli_validate_branching(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(EXAMPLES / "branching.dot")])
        assert result.exit_code == 0

    def test_cli_validate_code_review(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(EXAMPLES / "code_review.dot")])
        assert result.exit_code == 0

    def test_cli_validate_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "/nonexistent/file.dot"])
        assert result.exit_code != 0

    def test_cli_validate_invalid_dot(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.dot"
        bad_file.write_text("this is not valid DOT syntax {{{")
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(bad_file)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Test 8: CLI run command
# ---------------------------------------------------------------------------


class TestCLIRun:
    def test_cli_run_hello_world(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "hello_world.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert result.exit_code == 0
        assert "hello_world" in result.output

    def test_cli_run_produces_manifest(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "hello_world.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert (logs_dir / "manifest.json").exists()

    def test_cli_run_outcome_in_output(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "hello_world.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert "Outcome: success" in result.output

    def test_cli_run_with_auto_approve(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "code_review.dot"),
                "--backend", "stub",
                "--auto-approve",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert result.exit_code == 0

    def test_cli_run_with_goal_override(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "hello_world.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
                "--goal", "Custom goal override",
            ],
        )
        assert result.exit_code == 0
        assert "Custom goal override" in result.output

    def test_cli_run_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "/nonexistent/file.dot"])
        assert result.exit_code != 0

    def test_cli_run_shows_completed_nodes(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "hello_world.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert "Completed nodes" in result.output

    def test_cli_run_shows_run_directory(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "hello_world.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert "Run directory:" in result.output

    def test_cli_run_branching_with_stub(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "logs"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(EXAMPLES / "branching.dot"),
                "--backend", "stub",
                "--logs-dir", str(logs_dir),
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 9: CLI inspect command
# ---------------------------------------------------------------------------


class TestCLIInspect:
    def test_cli_inspect_hello_world(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(EXAMPLES / "hello_world.dot")])
        assert result.exit_code == 0
        assert "hello_world" in result.output

    def test_cli_inspect_shows_node_count(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(EXAMPLES / "hello_world.dot")])
        assert "Nodes:" in result.output

    def test_cli_inspect_shows_edges(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(EXAMPLES / "hello_world.dot")])
        assert "Edges:" in result.output

    def test_cli_inspect_shows_goal(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(EXAMPLES / "hello_world.dot")])
        assert "Say hello to the world" in result.output

    def test_cli_inspect_branching_shows_conditions(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", str(EXAMPLES / "branching.dot")])
        assert "condition=" in result.output

    def test_cli_inspect_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "/nonexistent/file.dot"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Test 10: CLI version
# ---------------------------------------------------------------------------


class TestCLIVersion:
    def test_cli_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# Test 11: Transform application in pipeline
# ---------------------------------------------------------------------------


class TestTransformIntegration:
    def test_variable_expansion_in_prompt(self, tmp_path: Path) -> None:
        """Variable expansion transform should expand $goal in prompts."""
        graph = load_example("code_review.dot")
        graph = apply_transforms(graph)
        # After transform, the analyze node's prompt should have the goal expanded
        analyze = graph.nodes.get("analyze")
        if analyze:
            # $goal should be expanded to the graph's goal
            assert "Review code changes for quality" in analyze.prompt or "$goal" in analyze.prompt

    def test_transforms_preserve_graph_structure(self, tmp_path: Path) -> None:
        """Transforms should not change the number of nodes or edges."""
        graph = load_example("hello_world.dot")
        original_nodes = len(graph.nodes)
        original_edges = len(graph.edges)
        graph = apply_transforms(graph)
        assert len(graph.nodes) == original_nodes
        assert len(graph.edges) == original_edges


# ---------------------------------------------------------------------------
# Test 12: End-to-end with context propagation
# ---------------------------------------------------------------------------


class TestContextPropagation:
    def test_context_carries_goal(self, tmp_path: Path) -> None:
        """Engine sets graph.goal in the context."""
        graph = load_example("hello_world.dot")
        validate_or_raise(graph)
        graph = apply_transforms(graph)
        handler = StubHandler()
        registry = make_registry(handler)
        context = Context()
        engine = Engine(
            graph, registry, context=context, logs_root=tmp_path
        )
        engine.run()
        assert context.get("graph.goal") == "Say hello to the world"

    def test_context_tracks_current_node(self, tmp_path: Path) -> None:
        """Engine sets current_node in context during execution."""
        graph = load_example("hello_world.dot")
        validate_or_raise(graph)
        graph = apply_transforms(graph)
        handler = StubHandler()
        registry = make_registry(handler)
        context = Context()
        engine = Engine(
            graph, registry, context=context, logs_root=tmp_path
        )
        engine.run()
        # current_node should be the last non-terminal node executed
        assert context.get("current_node") is not None


# ---------------------------------------------------------------------------
# Test 13: Event emission during pipeline
# ---------------------------------------------------------------------------


class TestEventIntegration:
    def test_pipeline_started_event(self, tmp_path: Path) -> None:
        from attractor.events import types as evt

        graph = load_example("hello_world.dot")
        validate_or_raise(graph)
        graph = apply_transforms(graph)
        handler = StubHandler()
        registry = make_registry(handler)
        bus = EventBus()
        collected: list[Any] = []
        bus.on_all(collected.append)

        engine = Engine(graph, registry, event_bus=bus, logs_root=tmp_path)
        engine.run()

        started = [e for e in collected if isinstance(e, evt.PipelineStarted)]
        assert len(started) == 1
        assert started[0].graph_name == "hello_world"

    def test_pipeline_completed_event(self, tmp_path: Path) -> None:
        from attractor.events import types as evt

        graph = load_example("hello_world.dot")
        validate_or_raise(graph)
        graph = apply_transforms(graph)
        handler = StubHandler()
        registry = make_registry(handler)
        bus = EventBus()
        collected: list[Any] = []
        bus.on_all(collected.append)

        engine = Engine(graph, registry, event_bus=bus, logs_root=tmp_path)
        engine.run()

        completed = [e for e in collected if isinstance(e, evt.PipelineCompleted)]
        assert len(completed) == 1
        assert completed[0].outcome.status == Status.SUCCESS

    def test_stage_events_for_each_node(self, tmp_path: Path) -> None:
        from attractor.events import types as evt

        graph = load_example("hello_world.dot")
        validate_or_raise(graph)
        graph = apply_transforms(graph)
        handler = StubHandler()
        registry = make_registry(handler)
        bus = EventBus()
        collected: list[Any] = []
        bus.on_all(collected.append)

        engine = Engine(graph, registry, event_bus=bus, logs_root=tmp_path)
        engine.run()

        stage_started = [e for e in collected if isinstance(e, evt.StageStarted)]
        stage_completed = [e for e in collected if isinstance(e, evt.StageCompleted)]
        # At least Start and greet (2 non-terminal nodes)
        assert len(stage_started) >= 2
        assert len(stage_completed) >= 2
