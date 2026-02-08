"""Event system: bus and event types for pipeline lifecycle."""

from attractor.events.bus import EventBus
from attractor.events.types import (
    CheckpointSaved,
    PipelineCompleted,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)

__all__ = [
    "EventBus",
    "CheckpointSaved",
    "PipelineCompleted",
    "PipelineFailed",
    "PipelineStarted",
    "StageCompleted",
    "StageFailed",
    "StageRetrying",
    "StageStarted",
]
