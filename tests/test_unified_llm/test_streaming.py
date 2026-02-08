"""Tests for unified_llm.types.streaming."""
from __future__ import annotations

import pytest

from unified_llm.types.enums import (
    ContentKind,
    FinishReason,
    Role,
    StreamEventType,
)
from unified_llm.types.content import ContentPart, ToolCallData
from unified_llm.types.messages import Message
from unified_llm.types.tools import ToolCall
from unified_llm.types.response import FinishReasonInfo, Response, Usage
from unified_llm.types.streaming import StreamAccumulator, StreamEvent


# ---------------------------------------------------------------------------
# StreamEvent construction
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_minimal(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_DELTA)
        assert evt.type == StreamEventType.TEXT_DELTA
        assert evt.delta is None
        assert evt.text_id is None
        assert evt.reasoning_delta is None
        assert evt.tool_call is None
        assert evt.finish_reason is None
        assert evt.usage is None
        assert evt.response is None
        assert evt.error is None
        assert evt.raw is None

    def test_text_delta(self) -> None:
        evt = StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            delta="hello",
            text_id="t0",
        )
        assert evt.delta == "hello"
        assert evt.text_id == "t0"

    def test_reasoning_delta(self) -> None:
        evt = StreamEvent(
            type=StreamEventType.REASONING_DELTA,
            reasoning_delta="thinking...",
        )
        assert evt.reasoning_delta == "thinking..."

    def test_frozen(self) -> None:
        evt = StreamEvent(type=StreamEventType.TEXT_DELTA)
        with pytest.raises(AttributeError):
            evt.delta = "nope"  # type: ignore[misc]

    def test_error_excluded_from_comparison(self) -> None:
        e1 = StreamEvent(type=StreamEventType.ERROR, error=ValueError("a"))
        e2 = StreamEvent(type=StreamEventType.ERROR, error=RuntimeError("b"))
        # error is compare=False
        assert e1 == e2

    def test_string_event_type(self) -> None:
        evt = StreamEvent(type="custom_provider_event")
        assert evt.type == "custom_provider_event"

    def test_with_raw(self) -> None:
        raw = {"custom": "data"}
        evt = StreamEvent(type=StreamEventType.PROVIDER_EVENT, raw=raw)
        assert evt.raw is raw


# ---------------------------------------------------------------------------
# StreamAccumulator — text
# ---------------------------------------------------------------------------


class TestStreamAccumulatorText:
    def test_empty(self) -> None:
        acc = StreamAccumulator()
        assert acc.text == ""

    def test_single_segment(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_START, text_id="t0"))
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello", text_id="t0"))
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta=" world", text_id="t0"))
        assert acc.text == "Hello world"

    def test_multiple_segments(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_START, text_id="a"))
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="First", text_id="a"))
        acc.process(StreamEvent(type=StreamEventType.TEXT_START, text_id="b"))
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Second", text_id="b"))
        assert "First" in acc.text
        assert "Second" in acc.text

    def test_text_delta_without_start(self) -> None:
        """TEXT_DELTA without prior TEXT_START should still accumulate."""
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="hi", text_id="t0"))
        assert acc.text == "hi"

    def test_text_delta_none_delta_ignored(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, text_id="t0"))
        assert acc.text == ""

    def test_no_text_id_defaults_to_empty(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="x"))
        assert acc.text == "x"


# ---------------------------------------------------------------------------
# StreamAccumulator — reasoning
# ---------------------------------------------------------------------------


class TestStreamAccumulatorReasoning:
    def test_empty(self) -> None:
        acc = StreamAccumulator()
        assert acc.reasoning == ""

    def test_accumulates(self) -> None:
        acc = StreamAccumulator()
        acc.process(
            StreamEvent(type=StreamEventType.REASONING_DELTA, reasoning_delta="step 1, ")
        )
        acc.process(
            StreamEvent(type=StreamEventType.REASONING_DELTA, reasoning_delta="step 2")
        )
        assert acc.reasoning == "step 1, step 2"

    def test_none_reasoning_delta_ignored(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.REASONING_DELTA))
        assert acc.reasoning == ""


# ---------------------------------------------------------------------------
# StreamAccumulator — tool calls
# ---------------------------------------------------------------------------


class TestStreamAccumulatorToolCalls:
    def test_empty(self) -> None:
        acc = StreamAccumulator()
        assert acc.tool_calls == []

    def test_accumulates(self) -> None:
        tc1 = ToolCall(id="c1", name="get_weather", arguments={"city": "NYC"})
        tc2 = ToolCall(id="c2", name="get_time", arguments={"tz": "UTC"})
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc1))
        acc.process(StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc2))
        assert acc.tool_calls == [tc1, tc2]

    def test_none_tool_call_ignored(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TOOL_CALL_END))
        assert acc.tool_calls == []

    def test_returns_copy(self) -> None:
        tc = ToolCall(id="c1", name="fn", arguments={})
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call=tc))
        calls = acc.tool_calls
        calls.clear()
        assert len(acc.tool_calls) == 1


# ---------------------------------------------------------------------------
# StreamAccumulator — response building
# ---------------------------------------------------------------------------


class TestStreamAccumulatorResponse:
    def test_none_before_finish(self) -> None:
        acc = StreamAccumulator()
        assert acc.response is None

    def test_builds_response(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_START, text_id="t0"))
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello", text_id="t0"))

        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        finish = FinishReasonInfo(reason=FinishReason.STOP)
        resp = Response(id="resp-1", model="gpt-4", provider="openai")

        acc.process(
            StreamEvent(
                type=StreamEventType.FINISH,
                usage=usage,
                finish_reason=finish,
                response=resp,
            )
        )

        result = acc.response
        assert result is not None
        assert result.id == "resp-1"
        assert result.model == "gpt-4"
        assert result.provider == "openai"
        assert result.finish_reason == finish
        assert result.usage == usage
        assert result.message.role == Role.ASSISTANT

    def test_response_includes_text_content(self) -> None:
        acc = StreamAccumulator()
        acc.process(StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hi", text_id="t0"))
        acc.process(
            StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
                response=Response(),
            )
        )
        result = acc.response
        assert result is not None
        text_parts = [p for p in result.message.content if p.kind == ContentKind.TEXT]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Hi"

    def test_response_includes_reasoning_content(self) -> None:
        acc = StreamAccumulator()
        acc.process(
            StreamEvent(type=StreamEventType.REASONING_DELTA, reasoning_delta="hmm")
        )
        acc.process(
            StreamEvent(
                type=StreamEventType.FINISH,
                finish_reason=FinishReasonInfo(reason=FinishReason.STOP),
                response=Response(),
            )
        )
        result = acc.response
        assert result is not None
        thinking_parts = [p for p in result.message.content if p.kind == ContentKind.THINKING]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].text == "hmm"
