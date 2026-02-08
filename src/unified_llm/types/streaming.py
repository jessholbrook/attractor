"""Streaming event types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unified_llm.types.enums import ContentKind, FinishReason, Role, StreamEventType
from unified_llm.types.content import ContentPart, ToolCallData
from unified_llm.types.messages import Message
from unified_llm.types.tools import ToolCall
from unified_llm.types.response import FinishReasonInfo, Response, Usage


@dataclass(frozen=True)
class StreamEvent:
    """A single event emitted during streaming."""

    type: StreamEventType | str
    delta: str | None = None
    text_id: str | None = None
    reasoning_delta: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: FinishReasonInfo | None = None
    usage: Usage | None = None
    response: Response | None = None
    error: Exception | None = field(default=None, compare=False, hash=False)
    raw: dict[str, Any] | None = None


class StreamAccumulator:
    """Collects :class:`StreamEvent` instances into a complete :class:`Response`."""

    def __init__(self) -> None:
        self._text_parts: dict[str, list[str]] = {}
        self._reasoning_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._usage: Usage | None = None
        self._finish_reason: FinishReasonInfo | None = None
        self._model: str = ""
        self._id: str = ""
        self._provider: str = ""

    def process(self, event: StreamEvent) -> None:
        """Process a stream event."""
        etype = event.type

        if etype == StreamEventType.TEXT_START:
            text_id = event.text_id or ""
            if text_id not in self._text_parts:
                self._text_parts[text_id] = []

        elif etype == StreamEventType.TEXT_DELTA:
            text_id = event.text_id or ""
            if text_id not in self._text_parts:
                self._text_parts[text_id] = []
            if event.delta is not None:
                self._text_parts[text_id].append(event.delta)

        elif etype == StreamEventType.REASONING_DELTA:
            if event.reasoning_delta is not None:
                self._reasoning_parts.append(event.reasoning_delta)

        elif etype == StreamEventType.TOOL_CALL_END:
            if event.tool_call is not None:
                self._tool_calls.append(event.tool_call)

        elif etype == StreamEventType.FINISH:
            if event.usage is not None:
                self._usage = event.usage
            if event.finish_reason is not None:
                self._finish_reason = event.finish_reason
            if event.response is not None:
                self._model = event.response.model
                self._id = event.response.id
                self._provider = event.response.provider

    @property
    def text(self) -> str:
        """All accumulated text, joined from all text segments."""
        parts: list[str] = []
        for chunks in self._text_parts.values():
            parts.append("".join(chunks))
        return "".join(parts)

    @property
    def reasoning(self) -> str:
        """All accumulated reasoning text."""
        return "".join(self._reasoning_parts)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All accumulated tool calls."""
        return list(self._tool_calls)

    @property
    def response(self) -> Response | None:
        """Build a Response from accumulated data. Returns None if no FINISH event received."""
        if self._finish_reason is None:
            return None

        text_content = self.text
        reasoning_content = self.reasoning or None

        content_parts: list[ContentPart] = []
        if text_content:
            content_parts.append(ContentPart(kind=ContentKind.TEXT, text=text_content))
        if reasoning_content:
            content_parts.append(ContentPart(kind=ContentKind.THINKING, text=reasoning_content))

        tool_call_data: list[ToolCallData] = []
        for tc in self._tool_calls:
            tool_call_data.append(
                ToolCallData(id=tc.id, name=tc.name, arguments=tc.arguments)
            )

        message = Message(
            role=Role.ASSISTANT,
            content=tuple(content_parts),
        )

        return Response(
            id=self._id,
            model=self._model,
            provider=self._provider,
            message=message,
            finish_reason=self._finish_reason,
            usage=self._usage or Usage(),
        )
