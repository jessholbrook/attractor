"""Tool definitions and tool call/result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from unified_llm.types.enums import ToolChoiceMode


@dataclass(frozen=True)
class Tool:
    """Definition of a tool that a model can call."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict, hash=False)
    execute: Callable[..., Any] | None = field(
        default=None, compare=False, hash=False,
    )


@dataclass(frozen=True)
class ToolChoice:
    """Controls how the model selects tools."""

    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    tool_name: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """A tool call made by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """The result of executing a tool."""

    tool_call_id: str
    content: str | dict[str, Any] | list[Any] = ""
    is_error: bool = False
