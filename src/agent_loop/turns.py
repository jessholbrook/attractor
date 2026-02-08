"""Turn types for the agent loop conversation history."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool call."""

    tool_call_id: str
    output: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class UserTurn:
    """User input turn."""

    content: str


@dataclass(frozen=True)
class AssistantTurn:
    """LLM response turn with optional tool calls and reasoning."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    response_id: str = ""


@dataclass(frozen=True)
class ToolResultsTurn:
    """Results from executing tool calls."""

    results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class SystemTurn:
    """System-injected message."""

    content: str


@dataclass(frozen=True)
class SteeringTurn:
    """Mid-task steering message injected by the host application."""

    content: str
    source: str = "host"


# Type alias for all turn types
Turn = UserTurn | AssistantTurn | ToolResultsTurn | SystemTurn | SteeringTurn
