from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from attractor.engine.engine import Engine, HandlerRegistry
from attractor.events.bus import EventBus
from attractor.handlers import create_default_registry
from attractor.interviewer.queue_interviewer import QueueInterviewer
from attractor.model.context import Context
from attractor.model.outcome import Outcome, Status

from wolverine.config import WolverineConfig
from wolverine.model.run import HealingRun, RunStatus
from wolverine.model.signal import RawSignal
from wolverine.pipeline.backend import StubWolverineBackend
from wolverine.pipeline.graph import build_wolverine_graph
from wolverine.pipeline.handlers import (
    ApplyToolHandler,
    ClassifyHandler,
    DeduplicateToolHandler,
    DiagnoseHandler,
    HealHandler,
    IngestToolHandler,
    ReviseHandler,
    ValidateHandler,
)
from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.store.repositories import RunRepository, SignalRepository


class WolverineRunner:
    """Orchestrator that wires adapters, pipeline graph, and the attractor engine."""

    def __init__(
        self,
        config: WolverineConfig,
        *,
        interviewer: QueueInterviewer | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self._db: Database | None = None
        self._signal_repo: SignalRepository | None = None
        self._run_repo: RunRepository | None = None
        self._interviewer = interviewer or QueueInterviewer()
        self._event_bus = event_bus or EventBus()

    def initialize(self) -> None:
        """Set up database, create tables, initialize repositories."""
        self._db = Database(self.config.db_path)
        self._db.connect()
        run_migrations(self._db)
        self._signal_repo = SignalRepository(self._db)
        self._run_repo = RunRepository(self._db)

    def build_registry(self) -> HandlerRegistry:
        """Build the handler registry with stub handlers for wolverine nodes.

        Uses ``create_default_registry`` for standard handlers (start, exit,
        conditional, codergen, wait.human), then registers tool functions that
        map to our custom tool node IDs.
        """
        # Tool functions for the ToolHandler -- keyed by node id.
        # Each receives a context snapshot dict and returns a dict of updates.
        ingest_tool = IngestToolHandler()
        dedup_tool = DeduplicateToolHandler()
        apply_tool = ApplyToolHandler()

        tool_registry: dict = {
            "ingest": lambda snap: {"ingested": True},
            "deduplicate": lambda snap: {"deduplicated": True},
            "apply": lambda snap: {"applied": True},
        }

        backend = StubWolverineBackend()

        registry = create_default_registry(
            codergen_backend=backend,
            interviewer=self._interviewer,
            tool_registry=tool_registry,
        )

        return registry

    def run_pipeline(self, signal: RawSignal) -> HealingRun:
        """Execute the full healing pipeline for a signal.

        1. Store signal in DB
        2. Create HealingRun record
        3. Build graph and registry
        4. Create Context with signal data
        5. Create Engine and run
        6. Update HealingRun with result
        """
        assert self._db is not None, "Runner not initialized -- call initialize() first"
        assert self._signal_repo is not None
        assert self._run_repo is not None

        # 1. Store signal
        self._signal_repo.create(signal)

        # 2. Create run
        run_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        run = HealingRun(
            id=run_id,
            signal_id=signal.id,
            status=RunStatus.INGESTING,
            started_at=now,
        )
        self._run_repo.create(run)

        # 3. Build graph and registry
        graph = build_wolverine_graph()
        registry = self.build_registry()

        # 4. Create context with signal data
        context = Context(
            initial={
                "signal_id": signal.id,
                "signal_title": signal.title,
                "signal_body": signal.body,
                "signal_kind": signal.kind.value,
                "signal_source": signal.source.value,
            }
        )

        # 5. Run engine
        logs_root = Path(self.config.log_dir) / run_id
        engine = Engine(
            graph,
            registry,
            context=context,
            event_bus=self._event_bus,
            logs_root=logs_root,
        )

        outcome = engine.run()

        # 6. Update run with result
        completed_at = datetime.now(timezone.utc).isoformat()
        if outcome.succeeded:
            self._run_repo.update_status(run_id, RunStatus.COMPLETED)
        else:
            self._run_repo.update_status(run_id, RunStatus.FAILED)
            self._run_repo.update_field(run_id, "error", outcome.failure_reason)
        self._run_repo.update_field(run_id, "completed_at", completed_at)

        # Return the updated run
        updated_run = self._run_repo.get(run_id)
        assert updated_run is not None
        return updated_run

    def close(self) -> None:
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None
