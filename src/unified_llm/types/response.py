"""Response types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unified_llm.types.enums import ContentKind, FinishReason, Role
from unified_llm.types.content import ContentPart, ToolCallData
from unified_llm.types.messages import Message


@dataclass(frozen=True)
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    raw: dict[str, Any] | None = None

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented

        def _sum_optional(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        input_t = self.input_tokens + other.input_tokens
        output_t = self.output_tokens + other.output_tokens
        return Usage(
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=input_t + output_t,
            reasoning_tokens=_sum_optional(self.reasoning_tokens, other.reasoning_tokens),
            cache_read_tokens=_sum_optional(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_sum_optional(self.cache_write_tokens, other.cache_write_tokens),
            raw=None,
        )


@dataclass(frozen=True)
class Warning:
    """A warning attached to a response."""

    message: str
    code: str | None = None


@dataclass(frozen=True)
class RateLimitInfo:
    """Rate limit information from the provider."""

    requests_remaining: int | None = None
    requests_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_limit: int | None = None
    reset_at: float | None = None


@dataclass(frozen=True)
class FinishReasonInfo:
    """Structured finish reason."""

    reason: FinishReason = FinishReason.STOP
    raw: str | None = None


@dataclass(frozen=True)
class Response:
    """A provider-agnostic LLM response."""

    id: str = ""
    model: str = ""
    provider: str = ""
    message: Message = field(default_factory=lambda: Message(role=Role.ASSISTANT))
    finish_reason: FinishReasonInfo = field(default_factory=FinishReasonInfo)
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] | None = None
    warnings: tuple[Warning, ...] = ()
    rate_limit: RateLimitInfo | None = None

    # --- Convenience properties ---

    @property
    def text(self) -> str:
        """Delegate to message.text."""
        return self.message.text

    @property
    def tool_calls(self) -> tuple[ToolCallData, ...]:
        """Extract all tool call data from the message."""
        return tuple(
            p.tool_call
            for p in self.message.content
            if p.kind == ContentKind.TOOL_CALL and p.tool_call is not None
        )

    @property
    def reasoning(self) -> str | None:
        """Concatenate all thinking content part texts. Returns None if none."""
        parts: list[str] = []
        for p in self.message.content:
            if p.kind == ContentKind.THINKING and p.thinking is not None:
                parts.append(p.thinking.text)
        return "".join(parts) if parts else None
