"""Checkpoint model: serialisable pipeline state for resume support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    """Snapshot of pipeline execution state that can be persisted and restored."""

    timestamp: str
    current_node: str
    completed_nodes: list[str] = field(default_factory=list)
    node_retries: dict[str, int] = field(default_factory=dict)
    context_values: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    # --- persistence ----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise to JSON and write to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": self.timestamp,
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "node_retries": self.node_retries,
            "context_values": self.context_values,
            "logs": self.logs,
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Deserialise a checkpoint from a JSON file at *path*."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            timestamp=data["timestamp"],
            current_node=data["current_node"],
            completed_nodes=data.get("completed_nodes", []),
            node_retries=data.get("node_retries", {}),
            context_values=data.get("context_values", {}),
            logs=data.get("logs", []),
        )

    # --- factory --------------------------------------------------------------

    @classmethod
    def create_now(
        cls,
        current_node: str,
        completed_nodes: list[str] | None = None,
        node_retries: dict[str, int] | None = None,
        context_values: dict[str, Any] | None = None,
        logs: list[str] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint stamped with the current UTC time."""
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            current_node=current_node,
            completed_nodes=completed_nodes or [],
            node_retries=node_retries or {},
            context_values=context_values or {},
            logs=logs or [],
        )
