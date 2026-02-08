"""Execution engine core: traverses the pipeline graph and runs node handlers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from attractor.engine.edge_selector import select_edge
from attractor.engine.retry import build_retry_policy
from attractor.events import types as events
from attractor.events.bus import EventBus
from attractor.model.checkpoint import Checkpoint
from attractor.model.context import Context
from attractor.model.graph import Graph, Node
from attractor.model.outcome import Outcome, Status


class Handler(Protocol):
    """Protocol for node handlers: must implement execute()."""

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: Path) -> Outcome: ...


class HandlerRegistry:
    """Maps node types and shapes to handler implementations."""

    SHAPE_TO_TYPE: dict[str, str] = {
        "Mdiamond": "start",
        "Msquare": "exit",
        "box": "codergen",
        "hexagon": "wait.human",
        "diamond": "conditional",
        "component": "parallel",
        "tripleoctagon": "parallel.fan_in",
        "parallelogram": "tool",
        "house": "stack.manager_loop",
    }

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._default: Handler | None = None

    def register(self, type_name: str, handler: Handler) -> None:
        """Register a handler for a named type."""
        self._handlers[type_name] = handler

    def set_default(self, handler: Handler) -> None:
        """Set the fallback handler used when no specific match is found."""
        self._default = handler

    def resolve(self, node: Node) -> Handler:
        """Resolve a handler for the given node.

        Resolution order:
        1. Explicit type attribute on the node
        2. Shape-based lookup via SHAPE_TO_TYPE
        3. Default handler
        """
        # 1. Explicit type attribute
        if node.type and node.type in self._handlers:
            return self._handlers[node.type]
        # 2. Shape-based resolution
        handler_type = self.SHAPE_TO_TYPE.get(node.shape, "")
        if handler_type and handler_type in self._handlers:
            return self._handlers[handler_type]
        # 3. Default
        if self._default:
            return self._default
        raise ValueError(
            f"No handler for node {node.id} (type={node.type!r}, shape={node.shape!r})"
        )


class Engine:
    """Pipeline execution engine: traverses the graph, executes handlers, manages state."""

    def __init__(
        self,
        graph: Graph,
        registry: HandlerRegistry,
        *,
        context: Context | None = None,
        event_bus: EventBus | None = None,
        logs_root: Path | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self.graph = graph
        self.registry = registry
        self.context = context or Context()
        self.event_bus = event_bus or EventBus()
        self.logs_root = logs_root or Path(
            f"attractor-runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        )
        self._checkpoint = checkpoint
        self._completed_nodes: list[str] = []
        self._node_outcomes: dict[str, Outcome] = {}
        self._node_retries: dict[str, int] = {}

    def run(self) -> Outcome:
        """Execute the full pipeline: find start, traverse graph, return final outcome."""
        self.logs_root.mkdir(parents=True, exist_ok=True)

        # Mirror graph attributes into context
        self.context.set("graph.goal", self.graph.goal)

        # Write manifest
        manifest = {
            "name": self.graph.name,
            "goal": self.graph.goal,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.logs_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self.event_bus.emit(events.PipelineStarted(graph_name=self.graph.name))

        # Restore from checkpoint if provided
        if self._checkpoint:
            self._restore_checkpoint(self._checkpoint)

        # Find start node
        start = self.graph.start_node()
        if not start:
            raise ValueError("No start node found in graph")

        current_node = start
        if self._checkpoint:
            # Resume: find the next node after the checkpointed one
            current_node = self._find_resume_node()

        last_outcome = Outcome(status=Status.SUCCESS)

        while True:
            node = current_node

            # Step 1: Check for terminal node
            if node.shape == "Msquare":
                gate_ok, failed_gate = self._check_goal_gates()
                if not gate_ok and failed_gate:
                    retry_target = self._get_retry_target(failed_gate)
                    if retry_target and retry_target in self.graph.nodes:
                        current_node = self.graph.nodes[retry_target]
                        continue
                    else:
                        self.event_bus.emit(
                            events.PipelineFailed(
                                graph_name=self.graph.name,
                                error=f"Goal gate unsatisfied on {failed_gate.id} and no retry target",
                            )
                        )
                        return Outcome(
                            status=Status.FAIL, failure_reason="Goal gate unsatisfied"
                        )
                # Pipeline complete
                self.event_bus.emit(
                    events.PipelineCompleted(graph_name=self.graph.name, outcome=last_outcome)
                )
                return last_outcome

            # Step 2: Execute node handler with retry
            self.event_bus.emit(events.StageStarted(node_id=node.id))
            self.context.set("current_node", node.id)

            retry_policy = build_retry_policy(node, self.graph)
            outcome = self._execute_with_retry(node, retry_policy)
            last_outcome = outcome

            # Step 3: Record completion
            self._completed_nodes.append(node.id)
            self._node_outcomes[node.id] = outcome

            self.event_bus.emit(events.StageCompleted(node_id=node.id, outcome=outcome))

            # Step 4: Apply context updates
            if outcome.context_updates:
                self.context.apply_updates(outcome.context_updates)
            self.context.set("outcome", outcome.status.value)
            if outcome.preferred_label:
                self.context.set("preferred_label", outcome.preferred_label)

            # Step 5: Save checkpoint
            cp = Checkpoint.create_now(
                current_node=node.id,
                completed_nodes=list(self._completed_nodes),
                node_retries=dict(self._node_retries),
                context_values=self.context.snapshot(),
            )
            cp_path = self.logs_root / "checkpoint.json"
            cp.save(cp_path)
            self.event_bus.emit(events.CheckpointSaved(node_id=node.id, path=str(cp_path)))

            # Step 6: Select next edge
            outgoing = self.graph.outgoing_edges(node.id)
            next_edge = select_edge(outgoing, outcome, self.context)

            if next_edge is None:
                if outcome.status == Status.FAIL:
                    self.event_bus.emit(
                        events.PipelineFailed(
                            graph_name=self.graph.name,
                            error=f"Stage {node.id} failed with no outgoing fail edge",
                        )
                    )
                    return outcome
                break

            # Step 7: Handle loop_restart
            if next_edge.loop_restart:
                # For now, just advance (full restart not implemented)
                pass

            # Step 8: Advance
            current_node = self.graph.nodes[next_edge.to_node]

        return last_outcome

    def _execute_with_retry(self, node: Node, retry_policy: Any) -> Outcome:
        """Execute a node handler with retry logic."""
        handler = self.registry.resolve(node)
        stage_dir = self.logs_root / node.id
        stage_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, retry_policy.max_attempts + 1):
            try:
                outcome = handler.execute(node, self.context, self.graph, stage_dir)
            except Exception as exc:
                if attempt < retry_policy.max_attempts:
                    delay = retry_policy.delay_for_attempt(attempt)
                    self.event_bus.emit(
                        events.StageRetrying(node_id=node.id, attempt=attempt, delay=delay)
                    )
                    time.sleep(delay)
                    continue
                return Outcome(status=Status.FAIL, failure_reason=str(exc))

            if outcome.status in (Status.SUCCESS, Status.PARTIAL_SUCCESS):
                self._node_retries.pop(node.id, None)
                self._write_status(stage_dir, outcome)
                return outcome

            if outcome.status == Status.RETRY:
                if attempt < retry_policy.max_attempts:
                    self._node_retries[node.id] = self._node_retries.get(node.id, 0) + 1
                    delay = retry_policy.delay_for_attempt(attempt)
                    self.event_bus.emit(
                        events.StageRetrying(node_id=node.id, attempt=attempt, delay=delay)
                    )
                    time.sleep(delay)
                    continue
                else:
                    if node.allow_partial:
                        outcome = Outcome(
                            status=Status.PARTIAL_SUCCESS,
                            notes="retries exhausted, partial accepted",
                        )
                    else:
                        outcome = Outcome(
                            status=Status.FAIL, failure_reason="max retries exceeded"
                        )
                    self._write_status(stage_dir, outcome)
                    return outcome

            if outcome.status == Status.FAIL:
                self._write_status(stage_dir, outcome)
                return outcome

        outcome = Outcome(status=Status.FAIL, failure_reason="max retries exceeded")
        self._write_status(stage_dir, outcome)
        return outcome

    def _check_goal_gates(self) -> tuple[bool, Node | None]:
        """Check all completed nodes with goal_gate=True passed."""
        for node_id, outcome in self._node_outcomes.items():
            node = self.graph.nodes[node_id]
            if node.goal_gate:
                if outcome.status not in (Status.SUCCESS, Status.PARTIAL_SUCCESS):
                    return False, node
        return True, None

    def _get_retry_target(self, node: Node) -> str | None:
        """Find the retry target for a failed goal gate node."""
        if node.retry_target:
            return node.retry_target
        if node.fallback_retry_target:
            return node.fallback_retry_target
        rt = self.graph.attributes.get("retry_target", "")
        if rt:
            return rt
        frt = self.graph.attributes.get("fallback_retry_target", "")
        if frt:
            return frt
        return None

    def _write_status(self, stage_dir: Path, outcome: Outcome) -> None:
        """Write a status.json file for the completed stage."""
        status_data = {
            "outcome": outcome.status.value,
            "preferred_next_label": outcome.preferred_label,
            "suggested_next_ids": outcome.suggested_next_ids,
            "context_updates": outcome.context_updates,
            "notes": outcome.notes,
        }
        (stage_dir / "status.json").write_text(json.dumps(status_data, indent=2, default=str))

    def _restore_checkpoint(self, cp: Checkpoint) -> None:
        """Restore engine state from a checkpoint."""
        self._completed_nodes = list(cp.completed_nodes)
        self._node_retries = dict(cp.node_retries)
        self.context.apply_updates(cp.context_values)

    def _find_resume_node(self) -> Node:
        """Find the node to resume from after checkpoint restore."""
        if not self._checkpoint or not self._completed_nodes:
            start = self.graph.start_node()
            if not start:
                raise ValueError("No start node")
            return start
        last = self._completed_nodes[-1]
        outgoing = self.graph.outgoing_edges(last)
        if outgoing:
            return self.graph.nodes[outgoing[0].to_node]
        start = self.graph.start_node()
        if not start:
            raise ValueError("No start node")
        return start
