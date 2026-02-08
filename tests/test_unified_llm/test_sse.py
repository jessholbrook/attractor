"""Tests for the SSE parser."""
from __future__ import annotations

import pytest

from unified_llm._sse import SSEEvent, parse_sse_lines


def _lines(*args: str) -> list[str]:
    """Helper to produce an iterator of lines."""
    return list(args)


# ---------------------------------------------------------------------------
# Basic events
# ---------------------------------------------------------------------------


def test_single_data_event() -> None:
    lines = iter(["data: hello", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].data == "hello"
    assert events[0].event == "message"


def test_event_with_type() -> None:
    lines = iter(["event: custom", "data: payload", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].event == "custom"
    assert events[0].data == "payload"


def test_multiple_events() -> None:
    lines = iter(["data: first", "", "data: second", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 2
    assert events[0].data == "first"
    assert events[1].data == "second"


# ---------------------------------------------------------------------------
# Multi-line data
# ---------------------------------------------------------------------------


def test_multi_line_data() -> None:
    lines = iter(["data: line1", "data: line2", "data: line3", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].data == "line1\nline2\nline3"


def test_data_with_empty_line_between() -> None:
    lines = iter(["data: line1", "data:", "data: line3", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].data == "line1\n\nline3"


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


def test_comment_lines_ignored() -> None:
    lines = iter([": this is a comment", "data: actual", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].data == "actual"


def test_only_comments_no_events() -> None:
    lines = iter([": comment 1", ": comment 2"])
    events = list(parse_sse_lines(lines))
    assert events == []


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_blank_line_without_data_no_event() -> None:
    lines = iter(["", ""])
    events = list(parse_sse_lines(lines))
    assert events == []


def test_empty_iterator() -> None:
    events = list(parse_sse_lines(iter([])))
    assert events == []


def test_event_without_trailing_blank_line() -> None:
    """Stream ends without a trailing blank line -- should still emit."""
    lines = iter(["data: trailing"])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].data == "trailing"


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------


def test_id_field() -> None:
    lines = iter(["id: 42", "data: msg", ""])
    events = list(parse_sse_lines(lines))
    assert events[0].id == "42"


def test_retry_field() -> None:
    lines = iter(["retry: 3000", "data: msg", ""])
    events = list(parse_sse_lines(lines))
    assert events[0].retry == 3000


def test_retry_non_integer_ignored() -> None:
    lines = iter(["retry: abc", "data: msg", ""])
    events = list(parse_sse_lines(lines))
    assert events[0].retry is None


def test_leading_space_stripped() -> None:
    """Per SSE spec, exactly one leading space after colon is stripped."""
    lines = iter(["data:  two spaces", ""])
    events = list(parse_sse_lines(lines))
    # One space stripped, one remains
    assert events[0].data == " two spaces"


def test_no_space_after_colon() -> None:
    lines = iter(["data:nospace", ""])
    events = list(parse_sse_lines(lines))
    assert events[0].data == "nospace"


# ---------------------------------------------------------------------------
# [DONE] handling (OpenAI)
# ---------------------------------------------------------------------------


def test_done_event() -> None:
    lines = iter(["data: [DONE]", ""])
    events = list(parse_sse_lines(lines))
    assert len(events) == 1
    assert events[0].data == "[DONE]"


# ---------------------------------------------------------------------------
# SSEEvent dataclass
# ---------------------------------------------------------------------------


def test_sse_event_defaults() -> None:
    e = SSEEvent()
    assert e.event == "message"
    assert e.data == ""
    assert e.id == ""
    assert e.retry is None


def test_sse_event_mutable() -> None:
    e = SSEEvent()
    e.event = "custom"
    e.data = "test"
    assert e.event == "custom"
    assert e.data == "test"
