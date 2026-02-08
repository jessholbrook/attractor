"""Session configuration and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SessionState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for a coding agent session."""

    max_turns: int = 0  # 0 = unlimited
    max_tool_rounds_per_input: int = 200
    default_command_timeout_ms: int = 10_000
    max_command_timeout_ms: int = 600_000
    reasoning_effort: str | None = None  # "low", "medium", "high"
    tool_output_limits: dict[str, int] = field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = 10
    max_subagent_depth: int = 1
