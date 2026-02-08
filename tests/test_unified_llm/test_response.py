"""Tests for unified_llm.types.response."""
from __future__ import annotations

from unified_llm.types.enums import ContentKind, FinishReason, Role
from unified_llm.types.content import ContentPart, ToolCallData, ThinkingData
from unified_llm.types.messages import Message
from unified_llm.types.response import (
    FinishReasonInfo,
    RateLimitInfo,
    Response,
    Usage,
    Warning,
)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

class TestUsageDefaults:
    """Test Usage default values."""

    def test_defaults(self) -> None:
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0
        assert u.reasoning_tokens is None
        assert u.cache_read_tokens is None
        assert u.cache_write_tokens is None
        assert u.raw is None

    def test_frozen(self) -> None:
        u = Usage()
        try:
            u.input_tokens = 5  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass


class TestUsageAddition:
    """Test Usage.__add__ — the critical operation."""

    def test_add_basic(self) -> None:
        a = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        b = Usage(input_tokens=5, output_tokens=15, total_tokens=20)
        c = a + b
        assert c.input_tokens == 15
        assert c.output_tokens == 35
        assert c.total_tokens == 50

    def test_add_total_is_recomputed(self) -> None:
        """total_tokens in result should be input + output, not sum of totals."""
        a = Usage(input_tokens=10, output_tokens=20, total_tokens=999)
        b = Usage(input_tokens=5, output_tokens=15, total_tokens=888)
        c = a + b
        assert c.total_tokens == 50  # 15 + 35, not 999 + 888

    def test_add_optional_both_none(self) -> None:
        a = Usage()
        b = Usage()
        c = a + b
        assert c.reasoning_tokens is None
        assert c.cache_read_tokens is None
        assert c.cache_write_tokens is None

    def test_add_optional_one_none(self) -> None:
        a = Usage(reasoning_tokens=10)
        b = Usage()
        c = a + b
        assert c.reasoning_tokens == 10

    def test_add_optional_other_none(self) -> None:
        a = Usage()
        b = Usage(cache_read_tokens=5)
        c = a + b
        assert c.cache_read_tokens == 5

    def test_add_optional_both_present(self) -> None:
        a = Usage(reasoning_tokens=10, cache_read_tokens=20, cache_write_tokens=5)
        b = Usage(reasoning_tokens=5, cache_read_tokens=10, cache_write_tokens=3)
        c = a + b
        assert c.reasoning_tokens == 15
        assert c.cache_read_tokens == 30
        assert c.cache_write_tokens == 8

    def test_add_raw_is_none(self) -> None:
        a = Usage(raw={"model": "gpt-4"})
        b = Usage(raw={"model": "gpt-4"})
        c = a + b
        assert c.raw is None

    def test_add_returns_not_implemented_for_wrong_type(self) -> None:
        u = Usage()
        result = u.__add__(42)  # type: ignore[arg-type]
        assert result is NotImplemented

    def test_add_chaining(self) -> None:
        a = Usage(input_tokens=1, output_tokens=2)
        b = Usage(input_tokens=3, output_tokens=4)
        c = Usage(input_tokens=5, output_tokens=6)
        total = a + b + c
        assert total.input_tokens == 9
        assert total.output_tokens == 12
        assert total.total_tokens == 21

    def test_add_zero_usage(self) -> None:
        a = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        b = Usage()
        c = a + b
        assert c.input_tokens == 10
        assert c.output_tokens == 20
        assert c.total_tokens == 30


# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------

class TestWarning:
    """Test Warning dataclass."""

    def test_warning_message_only(self) -> None:
        w = Warning(message="Heads up")
        assert w.message == "Heads up"
        assert w.code is None

    def test_warning_with_code(self) -> None:
        w = Warning(message="Deprecated", code="DEPRECATION")
        assert w.code == "DEPRECATION"

    def test_warning_frozen(self) -> None:
        w = Warning(message="x")
        try:
            w.message = "y"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# RateLimitInfo
# ---------------------------------------------------------------------------

class TestRateLimitInfo:
    """Test RateLimitInfo dataclass."""

    def test_defaults(self) -> None:
        rl = RateLimitInfo()
        assert rl.requests_remaining is None
        assert rl.requests_limit is None
        assert rl.tokens_remaining is None
        assert rl.tokens_limit is None
        assert rl.reset_at is None

    def test_with_values(self) -> None:
        rl = RateLimitInfo(
            requests_remaining=100,
            requests_limit=1000,
            tokens_remaining=50000,
            tokens_limit=100000,
            reset_at=1700000000.0,
        )
        assert rl.requests_remaining == 100
        assert rl.tokens_limit == 100000
        assert rl.reset_at == 1700000000.0


# ---------------------------------------------------------------------------
# FinishReasonInfo
# ---------------------------------------------------------------------------

class TestFinishReasonInfo:
    """Test FinishReasonInfo dataclass."""

    def test_default(self) -> None:
        fri = FinishReasonInfo()
        assert fri.reason == FinishReason.STOP
        assert fri.raw is None

    def test_with_reason(self) -> None:
        fri = FinishReasonInfo(reason=FinishReason.TOOL_CALLS, raw="tool_calls")
        assert fri.reason == FinishReason.TOOL_CALLS
        assert fri.raw == "tool_calls"

    def test_length_reason(self) -> None:
        fri = FinishReasonInfo(reason=FinishReason.LENGTH)
        assert fri.reason == FinishReason.LENGTH

    def test_content_filter_reason(self) -> None:
        fri = FinishReasonInfo(reason=FinishReason.CONTENT_FILTER)
        assert fri.reason == FinishReason.CONTENT_FILTER

    def test_frozen(self) -> None:
        fri = FinishReasonInfo()
        try:
            fri.reason = FinishReason.ERROR  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class TestResponseDefaults:
    """Test Response default values."""

    def test_defaults(self) -> None:
        r = Response()
        assert r.id == ""
        assert r.model == ""
        assert r.provider == ""
        assert r.message.role == Role.ASSISTANT
        assert r.message.content == ()
        assert r.finish_reason.reason == FinishReason.STOP
        assert r.usage.input_tokens == 0
        assert r.raw is None
        assert r.warnings == ()
        assert r.rate_limit is None

    def test_frozen(self) -> None:
        r = Response()
        try:
            r.id = "x"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass


class TestResponseText:
    """Test Response.text property."""

    def test_text_delegates_to_message(self) -> None:
        msg = Message.assistant("Hello there!")
        r = Response(message=msg)
        assert r.text == "Hello there!"

    def test_text_empty_when_no_content(self) -> None:
        r = Response()
        assert r.text == ""


class TestResponseToolCalls:
    """Test Response.tool_calls property."""

    def test_tool_calls_empty(self) -> None:
        r = Response()
        assert r.tool_calls == ()

    def test_tool_calls_extracts_from_message(self) -> None:
        tc1 = ToolCallData(id="tc1", name="search", arguments={"q": "test"})
        tc2 = ToolCallData(id="tc2", name="calc", arguments={"x": 1})
        msg = Message.assistant(tool_calls=[tc1, tc2])
        r = Response(message=msg)
        calls = r.tool_calls
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[1].name == "calc"

    def test_tool_calls_with_text_and_calls(self) -> None:
        tc = ToolCallData(id="tc1", name="fn", arguments={})
        msg = Message.assistant("thinking...", tool_calls=[tc])
        r = Response(message=msg)
        assert len(r.tool_calls) == 1
        assert r.text == "thinking..."


class TestResponseReasoning:
    """Test Response.reasoning property."""

    def test_reasoning_none_when_no_thinking_parts(self) -> None:
        r = Response(message=Message.assistant("Hello"))
        assert r.reasoning is None

    def test_reasoning_from_thinking_part(self) -> None:
        thinking_part = ContentPart(
            kind=ContentKind.THINKING,
            thinking=ThinkingData(text="Let me think about this."),
        )
        msg = Message(role=Role.ASSISTANT, content=(thinking_part,))
        r = Response(message=msg)
        assert r.reasoning == "Let me think about this."

    def test_reasoning_concatenates_multiple_parts(self) -> None:
        t1 = ContentPart(
            kind=ContentKind.THINKING,
            thinking=ThinkingData(text="Step 1. "),
        )
        t2 = ContentPart(
            kind=ContentKind.THINKING,
            thinking=ThinkingData(text="Step 2."),
        )
        text_part = ContentPart.of_text("Final answer.")
        msg = Message(role=Role.ASSISTANT, content=(t1, t2, text_part))
        r = Response(message=msg)
        assert r.reasoning == "Step 1. Step 2."
        assert r.text == "Final answer."

    def test_reasoning_none_for_empty_message(self) -> None:
        r = Response()
        assert r.reasoning is None


class TestResponseWithMetadata:
    """Test Response with warnings, rate limits, and usage."""

    def test_response_with_warnings(self) -> None:
        w = Warning(message="Low tokens", code="LOW_TOKENS")
        r = Response(warnings=(w,))
        assert len(r.warnings) == 1
        assert r.warnings[0].code == "LOW_TOKENS"

    def test_response_with_rate_limit(self) -> None:
        rl = RateLimitInfo(requests_remaining=50)
        r = Response(rate_limit=rl)
        assert r.rate_limit is not None
        assert r.rate_limit.requests_remaining == 50

    def test_response_with_usage(self) -> None:
        u = Usage(input_tokens=100, output_tokens=200, total_tokens=300)
        r = Response(usage=u)
        assert r.usage.input_tokens == 100
        assert r.usage.output_tokens == 200

    def test_response_full_construction(self) -> None:
        msg = Message.assistant("Hello!")
        r = Response(
            id="resp-123",
            model="gpt-4",
            provider="openai",
            message=msg,
            finish_reason=FinishReasonInfo(reason=FinishReason.STOP, raw="stop"),
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
            raw={"choices": [{"message": {"content": "Hello!"}}]},
            warnings=(Warning(message="test"),),
            rate_limit=RateLimitInfo(requests_remaining=99),
        )
        assert r.id == "resp-123"
        assert r.model == "gpt-4"
        assert r.provider == "openai"
        assert r.text == "Hello!"
        assert r.finish_reason.raw == "stop"
        assert r.raw is not None
