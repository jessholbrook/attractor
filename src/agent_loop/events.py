"""Event system for the agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventKind(Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_OUTPUT_DELTA = "tool_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"


# --- Event dataclasses ---


@dataclass(frozen=True)
class SessionStartEvent:
    session_id: str


@dataclass(frozen=True)
class SessionEndEvent:
    session_id: str
    reason: str = "completed"


@dataclass(frozen=True)
class UserInputEvent:
    content: str


@dataclass(frozen=True)
class AssistantTextStartEvent:
    pass


@dataclass(frozen=True)
class AssistantTextDeltaEvent:
    text: str


@dataclass(frozen=True)
class AssistantTextEndEvent:
    full_text: str
    reasoning: str = ""


@dataclass(frozen=True)
class ToolCallStartEvent:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutputDeltaEvent:
    tool_call_id: str
    delta: str


@dataclass(frozen=True)
class ToolCallEndEvent:
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class SteeringInjectedEvent:
    content: str
    source: str = "host"


@dataclass(frozen=True)
class TurnLimitEvent:
    turns_used: int
    max_turns: int


@dataclass(frozen=True)
class LoopDetectionEvent:
    message: str
    pattern: str = ""


@dataclass(frozen=True)
class ErrorEvent:
    error: str
    recoverable: bool = True


class EventEmitter:
    """Synchronous callback-based event emitter.

    Same API as attractor's EventBus for consistency.
    Events are dispatched synchronously in registration order.
    """

    def __init__(self) -> None:
        self._listeners: dict[type, list[Callable]] = {}
        self._global_listeners: list[Callable] = []

    def subscribe(self, event_type: type, callback: Callable) -> None:
        """Register a callback for a specific event type."""
        self._listeners.setdefault(event_type, []).append(callback)

    def on_all(self, callback: Callable) -> None:
        """Register a callback that receives every event."""
        self._global_listeners.append(callback)

    def emit(self, event: Any) -> None:
        """Dispatch event to all matching listeners."""
        for cb in self._global_listeners:
            cb(event)
        for cb in self._listeners.get(type(event), []):
            cb(event)
