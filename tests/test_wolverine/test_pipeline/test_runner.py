from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

import pytest

from attractor.events.bus import EventBus
from attractor.events.types import PipelineCompleted, PipelineStarted, StageCompleted
from attractor.interviewer.queue_interviewer import QueueInterviewer
from attractor.model.question import Answer, Option

from wolverine.config import WolverineConfig
from wolverine.model.run import RunStatus
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.runner import WolverineRunner


def _make_signal(
    title: str = "Test bug",
    body: str = "Something is broken",
    kind: SignalKind = SignalKind.MANUAL,
) -> RawSignal:
    return RawSignal(
        id=uuid.uuid4().hex,
        kind=kind,
        source=SignalSource.CLI,
        title=title,
        body=body,
        received_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_runner(
    tmp_path,
    *,
    interviewer: QueueInterviewer | None = None,
    event_bus: EventBus | None = None,
) -> WolverineRunner:
    config = WolverineConfig(
        db_path=str(tmp_path / "test.db"),
        log_dir=str(tmp_path / "logs"),
    )
    runner = WolverineRunner(
        config,
        interviewer=interviewer,
        event_bus=event_bus,
    )
    runner.initialize()
    return runner


class TestRunnerInit:
    def test_initializes_database(self, tmp_path) -> None:
        runner = _make_runner(tmp_path)
        assert runner._db is not None
        assert runner._signal_repo is not None
        assert runner._run_repo is not None
        runner.close()

    def test_close_cleans_up(self, tmp_path) -> None:
        runner = _make_runner(tmp_path)
        runner.close()
        assert runner._db is None

    def test_builds_registry(self, tmp_path) -> None:
        runner = _make_runner(tmp_path)
        registry = runner.build_registry()
        # Should not raise for any of our node types
        from wolverine.pipeline.graph import build_wolverine_graph

        graph = build_wolverine_graph()
        for node_id, node in graph.nodes.items():
            handler = registry.resolve(node)
            assert handler is not None, f"No handler for node {node_id}"
        runner.close()


class TestHappyPath:
    """Signal -> classify -> diagnose -> heal -> validate -> review(approve) -> apply -> exit."""

    def test_happy_path_completes(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        # Pre-load answer for review gate: "Approve"
        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            """Wait for the question then respond."""
            q = interviewer.pending_question(timeout=5.0)
            assert q is not None
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.status == RunStatus.COMPLETED
        runner.close()

    def test_happy_path_stores_signal(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        stored = runner._signal_repo.get(signal.id)
        assert stored is not None
        assert stored.title == signal.title
        runner.close()

    def test_happy_path_sets_completed_at(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.completed_at != ""
        runner.close()


class TestDuplicatePath:
    """Signal -> classify(is_duplicate=true) -> deduplicate -> exit."""

    def test_duplicate_path(self, tmp_path) -> None:
        """Override classify to set is_duplicate=true and verify dedup path."""
        from attractor.model.context import Context
        from attractor.model.outcome import Outcome, Status
        from wolverine.pipeline.graph import build_wolverine_graph

        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        # We need to override the codergen handler for 'classify' to set is_duplicate=true.
        # The classify node is type "box" (codergen). The StubWolverineBackend returns "Stub response".
        # But the classify handler's context_updates are set by the CodergenHandler, which just puts
        # the response in context under "classify.response". The conditional node is_duplicate then
        # checks context.is_duplicate.
        #
        # To make the duplicate path work, we need to pre-set is_duplicate=true in context
        # before the conditional node evaluates. We'll do this by modifying the tool registry
        # for the ingest tool to also set is_duplicate.

        # Actually, let's build and run our own engine with a custom context
        from attractor.engine.engine import Engine
        from attractor.handlers import create_default_registry
        from wolverine.pipeline.backend import StubWolverineBackend
        from pathlib import Path as P

        graph = build_wolverine_graph()

        tool_registry = {
            "ingest": lambda snap: {"ingested": True, "is_duplicate": "true"},
            "deduplicate": lambda snap: {"deduplicated": True},
            "apply": lambda snap: {"applied": True},
        }
        backend = StubWolverineBackend()
        registry = create_default_registry(
            codergen_backend=backend,
            interviewer=interviewer,
            tool_registry=tool_registry,
        )

        context = Context(initial={
            "signal_id": signal.id,
            "signal_title": signal.title,
        })

        logs_root = P(str(tmp_path)) / "dup_test"
        engine = Engine(graph, registry, context=context, logs_root=logs_root)
        outcome = engine.run()

        assert outcome.succeeded
        # The context should show deduplicated = True
        assert context.get("deduplicated") is True
        runner.close()


class TestRejectPath:
    """Signal -> ... -> review(reject) -> exit."""

    def test_reject_path(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="2", label="Reject"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        # Even with reject, the pipeline completes (exits normally)
        assert run.status == RunStatus.COMPLETED
        runner.close()


class TestRequestChangesPath:
    """Signal -> ... -> review(request_changes) -> revise -> validate -> review(approve) -> apply -> exit."""

    def test_request_changes_then_approve(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answers = [
            Answer(selected_option=Option(key="1", label="Request Changes")),
            Answer(selected_option=Option(key="0", label="Approve")),
        ]
        answer_idx = 0

        def feed_answers():
            nonlocal answer_idx
            for ans in answers:
                q = interviewer.pending_question(timeout=5.0)
                assert q is not None
                interviewer.respond(ans)

        thread = threading.Thread(target=feed_answers, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.status == RunStatus.COMPLETED
        runner.close()


class TestEventBus:
    def test_pipeline_events_emitted(self, tmp_path) -> None:
        events_received: list = []
        event_bus = EventBus()
        event_bus.on_all(lambda e: events_received.append(e))

        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer, event_bus=event_bus)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        event_types = [type(e).__name__ for e in events_received]
        assert "PipelineStarted" in event_types
        assert "PipelineCompleted" in event_types
        assert "StageCompleted" in event_types
        runner.close()


class TestRunnerPipelineRecord:
    def test_run_has_signal_id(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.signal_id == signal.id
        runner.close()

    def test_run_id_is_nonempty(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.id != ""
        runner.close()

    def test_run_started_at(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        def feed_answer():
            q = interviewer.pending_question(timeout=5.0)
            interviewer.respond(answer)

        thread = threading.Thread(target=feed_answer, daemon=True)
        thread.start()

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.started_at != ""
        runner.close()
