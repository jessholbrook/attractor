"""Generation result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unified_llm.types.enums import FinishReason
from unified_llm.types.tools import ToolCall, ToolResult
from unified_llm.types.response import FinishReasonInfo, Response, Usage, Warning


@dataclass(frozen=True)
class StepResult:
    """Result of a single generation step (one round-trip to the provider)."""

    text: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    finish_reason: FinishReasonInfo = field(default_factory=FinishReasonInfo)
    usage: Usage = field(default_factory=Usage)
    response: Response = field(default_factory=Response)
    warnings: tuple[Warning, ...] = ()


@dataclass(frozen=True)
class GenerateResult:
    """Result of a complete generation (potentially multiple steps with tool use)."""

    text: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    finish_reason: FinishReasonInfo = field(default_factory=FinishReasonInfo)
    usage: Usage = field(default_factory=Usage)
    total_usage: Usage = field(default_factory=Usage)
    steps: tuple[StepResult, ...] = ()
    response: Response = field(default_factory=Response)
    output: Any = None
