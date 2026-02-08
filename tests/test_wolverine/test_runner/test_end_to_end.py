"""End-to-end integration tests for the Wolverine pipeline."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from attractor.engine.engine import Engine, HandlerRegistry
from attractor.events.bus import EventBus
from attractor.events.types import PipelineCompleted, PipelineStarted, StageCompleted
from attractor.handlers import create_default_registry
from attractor.interviewer.queue_interviewer import QueueInterviewer
from attractor.model.context import Context
from attractor.model.outcome import Outcome, Status
from attractor.model.question import Answer, Option

from wolverine.config import WolverineConfig
from wolverine.model.run import HealingRun, RunStatus
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.pipeline.backend import StubWolverineBackend
from wolverine.pipeline.graph import build_wolverine_graph
from wolverine.runner import WolverineRunner
from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.store.repositories import RunRepository, SignalRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    database = Database(":memory:")
    database.connect()
    run_migrations(database)
    yield database
    database.close()


@pytest.fixture
def sample_signal():
    return RawSignal(
        id="sig-001",
        kind=SignalKind.USER_FEEDBACK,
        source=SignalSource.API,
        title="Help center missing video docs",
        body="I can't figure out how to use the video feature and there is nothing in the Help Center about it.",
        received_at="2025-01-15T10:00:00Z",
    )


def _make_signal(
    title: str = "Test bug",
    body: str = "Something is broken",
    kind: SignalKind = SignalKind.MANUAL,
    sig_id: str | None = None,
) -> RawSignal:
    return RawSignal(
        id=sig_id or uuid.uuid4().hex[:12],
        kind=kind,
        source=SignalSource.CLI,
        title=title,
        body=body,
        received_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_runner(
    tmp_path: Path,
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


def _feed_answers(
    interviewer: QueueInterviewer,
    answers: list[Answer],
    timeout: float = 5.0,
) -> threading.Thread:
    """Start a background thread that feeds answers to the interviewer."""

    def _worker():
        for ans in answers:
            q = interviewer.pending_question(timeout=timeout)
            assert q is not None, "Expected a question but got None (timeout)"
            interviewer.respond(ans)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Happy path: approve
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Signal -> classify -> diagnose -> heal -> validate -> review(approve) -> apply -> exit."""

    def test_happy_path_completes(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.status == RunStatus.COMPLETED
        runner.close()

    def test_happy_path_stores_signal(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        runner.run_pipeline(signal)
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
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.completed_at != ""
        runner.close()

    def test_happy_path_run_has_signal_id(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.signal_id == signal.id
        runner.close()

    def test_happy_path_run_id_is_nonempty(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.id != ""
        runner.close()

    def test_happy_path_run_started_at(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.started_at != ""
        runner.close()

    def test_happy_path_with_sample_signal(self, tmp_path, sample_signal) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(sample_signal)
        thread.join(timeout=10.0)

        assert run.status == RunStatus.COMPLETED
        stored = runner._signal_repo.get(sample_signal.id)
        assert stored is not None
        assert stored.title == "Help center missing video docs"
        runner.close()


# ---------------------------------------------------------------------------
# Duplicate path
# ---------------------------------------------------------------------------


class TestDuplicatePath:
    """Signal -> classify(is_duplicate=true) -> deduplicate -> exit."""

    def test_duplicate_path_completes(self, tmp_path) -> None:
        """Override ingest tool to set is_duplicate=true, verify dedup path."""
        interviewer = QueueInterviewer(timeout=5.0)

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

        signal = _make_signal()
        context = Context(
            initial={
                "signal_id": signal.id,
                "signal_title": signal.title,
            }
        )

        logs_root = Path(str(tmp_path)) / "dup_test"
        engine = Engine(graph, registry, context=context, logs_root=logs_root)
        outcome = engine.run()

        assert outcome.succeeded
        assert context.get("deduplicated") is True

    def test_duplicate_does_not_reach_diagnose(self, tmp_path) -> None:
        """Ensure the duplicate path skips diagnosis and healing."""
        interviewer = QueueInterviewer(timeout=5.0)

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

        context = Context(initial={"signal_id": "test", "signal_title": "test"})
        logs_root = Path(str(tmp_path)) / "dup_test2"
        engine = Engine(graph, registry, context=context, logs_root=logs_root)
        outcome = engine.run()

        assert outcome.succeeded
        # root_cause would be set if diagnose ran
        assert context.get("root_cause") is None


# ---------------------------------------------------------------------------
# Reject path
# ---------------------------------------------------------------------------


class TestRejectPath:
    """Signal -> ... -> review(reject) -> exit."""

    def test_reject_path_completes(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="2", label="Reject"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.status == RunStatus.COMPLETED
        runner.close()

    def test_reject_does_not_apply(self, tmp_path) -> None:
        """On reject, the pipeline exits without applying."""
        interviewer = QueueInterviewer(timeout=5.0)

        graph = build_wolverine_graph()

        tool_registry = {
            "ingest": lambda snap: {"ingested": True},
            "deduplicate": lambda snap: {"deduplicated": True},
            "apply": lambda snap: {"applied": True},
        }
        backend = StubWolverineBackend()
        registry = create_default_registry(
            codergen_backend=backend,
            interviewer=interviewer,
            tool_registry=tool_registry,
        )

        answer = Answer(selected_option=Option(key="2", label="Reject"))
        thread = _feed_answers(interviewer, [answer])

        context = Context(initial={"signal_id": "test", "signal_title": "test"})
        logs_root = Path(str(tmp_path)) / "reject_test"
        engine = Engine(graph, registry, context=context, logs_root=logs_root)
        outcome = engine.run()
        thread.join(timeout=10.0)

        assert outcome.succeeded
        # apply would set applied=True
        assert context.get("applied") is None


# ---------------------------------------------------------------------------
# Request changes -> approve
# ---------------------------------------------------------------------------


class TestRequestChangesPath:
    """Signal -> ... -> review(request_changes) -> revise -> validate -> review(approve) -> exit."""

    def test_request_changes_then_approve(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)
        signal = _make_signal()

        answers = [
            Answer(selected_option=Option(key="1", label="Request Changes")),
            Answer(selected_option=Option(key="0", label="Approve")),
        ]
        thread = _feed_answers(interviewer, answers)

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        assert run.status == RunStatus.COMPLETED
        runner.close()

    def test_request_changes_runs_revision(self, tmp_path) -> None:
        """Verify that the revise node is executed when changes are requested."""
        interviewer = QueueInterviewer(timeout=5.0)

        graph = build_wolverine_graph()

        tool_registry = {
            "ingest": lambda snap: {"ingested": True},
            "deduplicate": lambda snap: {"deduplicated": True},
            "apply": lambda snap: {"applied": True},
        }
        backend = StubWolverineBackend()
        registry = create_default_registry(
            codergen_backend=backend,
            interviewer=interviewer,
            tool_registry=tool_registry,
        )

        answers = [
            Answer(selected_option=Option(key="1", label="Request Changes")),
            Answer(selected_option=Option(key="0", label="Approve")),
        ]
        thread = _feed_answers(interviewer, answers)

        context = Context(initial={"signal_id": "test", "signal_title": "test"})
        logs_root = Path(str(tmp_path)) / "changes_test"
        engine = Engine(graph, registry, context=context, logs_root=logs_root)
        outcome = engine.run()
        thread.join(timeout=10.0)

        assert outcome.succeeded
        # The CodergenHandler stores the revise output in "revise.response".
        # Its presence proves the revise node executed.
        assert context.get("revise.response") is not None


# ---------------------------------------------------------------------------
# Runner initialization
# ---------------------------------------------------------------------------


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
        graph = build_wolverine_graph()
        for node_id, node in graph.nodes.items():
            handler = registry.resolve(node)
            assert handler is not None, f"No handler for node {node_id}"
        runner.close()

    def test_registry_has_tool_handler(self, tmp_path) -> None:
        runner = _make_runner(tmp_path)
        registry = runner.build_registry()
        graph = build_wolverine_graph()
        # The ingest node is type="tool", should resolve
        ingest_node = graph.nodes["ingest"]
        handler = registry.resolve(ingest_node)
        assert handler is not None
        runner.close()

    def test_registry_has_codergen_handler(self, tmp_path) -> None:
        runner = _make_runner(tmp_path)
        registry = runner.build_registry()
        graph = build_wolverine_graph()
        # The classify node is shape="box" (codergen)
        classify_node = graph.nodes["classify"]
        handler = registry.resolve(classify_node)
        assert handler is not None
        runner.close()


# ---------------------------------------------------------------------------
# Multiple signals
# ---------------------------------------------------------------------------


class TestMultipleSignals:
    def test_two_signals_complete_independently(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)

        signal1 = _make_signal(title="Bug 1", body="First bug")
        signal2 = _make_signal(title="Bug 2", body="Second bug")

        answer1 = Answer(selected_option=Option(key="0", label="Approve"))

        thread1 = _feed_answers(interviewer, [answer1])
        run1 = runner.run_pipeline(signal1)
        thread1.join(timeout=10.0)

        answer2 = Answer(selected_option=Option(key="0", label="Approve"))
        thread2 = _feed_answers(interviewer, [answer2])
        run2 = runner.run_pipeline(signal2)
        thread2.join(timeout=10.0)

        assert run1.status == RunStatus.COMPLETED
        assert run2.status == RunStatus.COMPLETED
        assert run1.id != run2.id
        runner.close()

    def test_signals_stored_separately(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)

        signal1 = _make_signal(title="Bug A", sig_id="sig-a")
        signal2 = _make_signal(title="Bug B", sig_id="sig-b")

        answer = Answer(selected_option=Option(key="0", label="Approve"))

        thread1 = _feed_answers(interviewer, [answer])
        runner.run_pipeline(signal1)
        thread1.join(timeout=10.0)

        answer2 = Answer(selected_option=Option(key="0", label="Approve"))
        thread2 = _feed_answers(interviewer, [answer2])
        runner.run_pipeline(signal2)
        thread2.join(timeout=10.0)

        stored1 = runner._signal_repo.get("sig-a")
        stored2 = runner._signal_repo.get("sig-b")
        assert stored1 is not None
        assert stored2 is not None
        assert stored1.title == "Bug A"
        assert stored2.title == "Bug B"
        runner.close()


# ---------------------------------------------------------------------------
# Signal with metadata
# ---------------------------------------------------------------------------


class TestSignalMetadata:
    def test_metadata_preserved(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)

        signal = RawSignal(
            id="sig-meta",
            kind=SignalKind.USER_FEEDBACK,
            source=SignalSource.API,
            title="Bug with metadata",
            body="Details here",
            received_at="2025-01-15T10:00:00Z",
            metadata={"source_url": "https://example.com", "user_id": "u-123"},
        )

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        stored = runner._signal_repo.get("sig-meta")
        assert stored is not None
        assert stored.metadata["source_url"] == "https://example.com"
        assert stored.metadata["user_id"] == "u-123"
        runner.close()

    def test_empty_metadata_ok(self, tmp_path) -> None:
        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer)

        signal = _make_signal(sig_id="sig-empty-meta")

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        run = runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        stored = runner._signal_repo.get("sig-empty-meta")
        assert stored is not None
        assert stored.metadata == {}
        runner.close()


# ---------------------------------------------------------------------------
# Event bus integration
# ---------------------------------------------------------------------------


class TestEventBusIntegration:
    def test_pipeline_events_emitted(self, tmp_path) -> None:
        events_received: list = []
        event_bus = EventBus()
        event_bus.on_all(lambda e: events_received.append(e))

        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer, event_bus=event_bus)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        event_types = [type(e).__name__ for e in events_received]
        assert "PipelineStarted" in event_types
        assert "PipelineCompleted" in event_types
        assert "StageCompleted" in event_types
        runner.close()

    def test_stage_completed_events_for_each_node(self, tmp_path) -> None:
        events_received: list = []
        event_bus = EventBus()
        event_bus.on_all(lambda e: events_received.append(e))

        interviewer = QueueInterviewer(timeout=5.0)
        runner = _make_runner(tmp_path, interviewer=interviewer, event_bus=event_bus)
        signal = _make_signal()

        answer = Answer(selected_option=Option(key="0", label="Approve"))
        thread = _feed_answers(interviewer, [answer])

        runner.run_pipeline(signal)
        thread.join(timeout=10.0)

        stage_events = [e for e in events_received if type(e).__name__ == "StageCompleted"]
        node_ids = [e.node_id for e in stage_events]
        # The happy path visits: Start, ingest, classify, is_duplicate, diagnose,
        # heal, validate, review_gate, apply
        assert "Start" in node_ids
        assert "ingest" in node_ids
        assert "classify" in node_ids
        runner.close()
