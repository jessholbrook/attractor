"""Request types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from unified_llm.types.messages import Message
from unified_llm.types.tools import Tool, ToolChoice


@dataclass(frozen=True)
class ResponseFormat:
    """Desired response format."""

    type: str = "text"
    json_schema: dict[str, Any] | None = None
    strict: bool = False


@dataclass(frozen=True)
class Request:
    """A provider-agnostic LLM request."""

    model: str
    messages: tuple[Message, ...] = ()
    provider: str | None = None
    tools: tuple[Tool, ...] | None = None
    tool_choice: ToolChoice | None = None
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    reasoning_effort: str | None = None
    metadata: dict[str, str] | None = None
    provider_options: dict[str, Any] | None = None
