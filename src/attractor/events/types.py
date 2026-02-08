"""Event types emitted during pipeline execution."""

from dataclasses import dataclass

from attractor.model.outcome import Outcome


@dataclass(frozen=True)
class PipelineStarted:
    graph_name: str


@dataclass(frozen=True)
class PipelineCompleted:
    graph_name: str
    outcome: Outcome


@dataclass(frozen=True)
class PipelineFailed:
    graph_name: str
    error: str


@dataclass(frozen=True)
class StageStarted:
    node_id: str


@dataclass(frozen=True)
class StageCompleted:
    node_id: str
    outcome: Outcome


@dataclass(frozen=True)
class StageFailed:
    node_id: str
    error: str
    will_retry: bool


@dataclass(frozen=True)
class StageRetrying:
    node_id: str
    attempt: int
    delay: float


@dataclass(frozen=True)
class CheckpointSaved:
    node_id: str
    path: str
