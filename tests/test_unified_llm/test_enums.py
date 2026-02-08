"""Tests for unified LLM enumeration types."""
from __future__ import annotations

import pytest

from unified_llm.types.enums import (
    ContentKind,
    FinishReason,
    Role,
    StreamEventType,
    ToolChoiceMode,
)


class TestRole:
    def test_has_5_members(self) -> None:
        assert len(Role) == 5

    def test_system_value(self) -> None:
        assert Role.SYSTEM == "system"
        assert Role.SYSTEM.value == "system"

    def test_user_value(self) -> None:
        assert Role.USER == "user"

    def test_assistant_value(self) -> None:
        assert Role.ASSISTANT == "assistant"

    def test_tool_value(self) -> None:
        assert Role.TOOL == "tool"

    def test_developer_value(self) -> None:
        assert Role.DEVELOPER == "developer"

    def test_is_str_subclass(self) -> None:
        assert isinstance(Role.USER, str)

    def test_can_be_used_as_dict_key(self) -> None:
        d = {Role.USER: "hello"}
        assert d["user"] == "hello"


class TestContentKind:
    def test_has_8_members(self) -> None:
        assert len(ContentKind) == 8

    def test_all_values(self) -> None:
        expected = {
            "text",
            "image",
            "audio",
            "document",
            "tool_call",
            "tool_result",
            "thinking",
            "redacted_thinking",
        }
        assert {k.value for k in ContentKind} == expected

    def test_text_value(self) -> None:
        assert ContentKind.TEXT == "text"

    def test_redacted_thinking_value(self) -> None:
        assert ContentKind.REDACTED_THINKING == "redacted_thinking"


class TestFinishReason:
    def test_has_6_members(self) -> None:
        assert len(FinishReason) == 6

    def test_all_values(self) -> None:
        expected = {"stop", "length", "tool_calls", "content_filter", "error", "other"}
        assert {r.value for r in FinishReason} == expected

    def test_stop_value(self) -> None:
        assert FinishReason.STOP == "stop"

    def test_tool_calls_value(self) -> None:
        assert FinishReason.TOOL_CALLS == "tool_calls"

    def test_content_filter_value(self) -> None:
        assert FinishReason.CONTENT_FILTER == "content_filter"


class TestStreamEventType:
    def test_has_13_members(self) -> None:
        assert len(StreamEventType) == 13

    def test_all_values(self) -> None:
        expected = {
            "stream_start",
            "text_start",
            "text_delta",
            "text_end",
            "reasoning_start",
            "reasoning_delta",
            "reasoning_end",
            "tool_call_start",
            "tool_call_delta",
            "tool_call_end",
            "finish",
            "error",
            "provider_event",
        }
        assert {e.value for e in StreamEventType} == expected

    def test_stream_start_value(self) -> None:
        assert StreamEventType.STREAM_START == "stream_start"

    def test_provider_event_value(self) -> None:
        assert StreamEventType.PROVIDER_EVENT == "provider_event"

    def test_finish_value(self) -> None:
        assert StreamEventType.FINISH == "finish"


class TestToolChoiceMode:
    def test_has_4_members(self) -> None:
        assert len(ToolChoiceMode) == 4

    def test_all_values(self) -> None:
        expected = {"auto", "none", "required", "named"}
        assert {m.value for m in ToolChoiceMode} == expected

    def test_auto_value(self) -> None:
        assert ToolChoiceMode.AUTO == "auto"

    def test_named_value(self) -> None:
        assert ToolChoiceMode.NAMED == "named"

    def test_none_value(self) -> None:
        assert ToolChoiceMode.NONE == "none"

    def test_required_value(self) -> None:
        assert ToolChoiceMode.REQUIRED == "required"
