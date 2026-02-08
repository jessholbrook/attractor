"""Enumeration types for the unified LLM client."""
from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Role of a message participant."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentKind(StrEnum):
    """Discriminator for content part types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"


class FinishReason(StrEnum):
    """Why the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    OTHER = "other"


class StreamEventType(StrEnum):
    """Types of events emitted during streaming."""

    STREAM_START = "stream_start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    FINISH = "finish"
    ERROR = "error"
    PROVIDER_EVENT = "provider_event"


class ToolChoiceMode(StrEnum):
    """How the model should choose which tools to call."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"
