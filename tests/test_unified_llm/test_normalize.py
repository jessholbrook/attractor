"""Tests for response normalization helpers."""
from __future__ import annotations

import pytest

from unified_llm._normalize import (
    classify_error_message,
    extract_rate_limit,
    map_finish_reason,
)
from unified_llm.types.enums import FinishReason


# ---------------------------------------------------------------------------
# map_finish_reason — OpenAI
# ---------------------------------------------------------------------------


def test_openai_stop() -> None:
    info = map_finish_reason("stop", "openai")
    assert info.reason == FinishReason.STOP
    assert info.raw == "stop"


def test_openai_length() -> None:
    info = map_finish_reason("length", "openai")
    assert info.reason == FinishReason.LENGTH


def test_openai_tool_calls() -> None:
    info = map_finish_reason("tool_calls", "openai")
    assert info.reason == FinishReason.TOOL_CALLS


def test_openai_content_filter() -> None:
    info = map_finish_reason("content_filter", "openai")
    assert info.reason == FinishReason.CONTENT_FILTER


# ---------------------------------------------------------------------------
# map_finish_reason — Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_end_turn() -> None:
    info = map_finish_reason("end_turn", "anthropic")
    assert info.reason == FinishReason.STOP
    assert info.raw == "end_turn"


def test_anthropic_stop_sequence() -> None:
    info = map_finish_reason("stop_sequence", "anthropic")
    assert info.reason == FinishReason.STOP


def test_anthropic_max_tokens() -> None:
    info = map_finish_reason("max_tokens", "anthropic")
    assert info.reason == FinishReason.LENGTH


def test_anthropic_tool_use() -> None:
    info = map_finish_reason("tool_use", "anthropic")
    assert info.reason == FinishReason.TOOL_CALLS


# ---------------------------------------------------------------------------
# map_finish_reason — Gemini
# ---------------------------------------------------------------------------


def test_gemini_stop() -> None:
    info = map_finish_reason("STOP", "gemini")
    assert info.reason == FinishReason.STOP


def test_gemini_max_tokens() -> None:
    info = map_finish_reason("MAX_TOKENS", "gemini")
    assert info.reason == FinishReason.LENGTH


def test_gemini_safety() -> None:
    info = map_finish_reason("SAFETY", "gemini")
    assert info.reason == FinishReason.CONTENT_FILTER


def test_gemini_recitation() -> None:
    info = map_finish_reason("RECITATION", "gemini")
    assert info.reason == FinishReason.CONTENT_FILTER


# ---------------------------------------------------------------------------
# map_finish_reason — Unknown
# ---------------------------------------------------------------------------


def test_unknown_reason() -> None:
    info = map_finish_reason("weird_reason", "openai")
    assert info.reason == FinishReason.OTHER
    assert info.raw == "weird_reason"


def test_unknown_provider() -> None:
    info = map_finish_reason("stop", "unknown_provider")
    assert info.reason == FinishReason.OTHER
    assert info.raw == "stop"


# ---------------------------------------------------------------------------
# extract_rate_limit
# ---------------------------------------------------------------------------


def test_extract_rate_limit_full() -> None:
    headers = {
        "x-ratelimit-remaining-requests": "99",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-tokens": "9000",
        "x-ratelimit-limit-tokens": "10000",
        "x-ratelimit-reset": "1700000000.0",
    }
    info = extract_rate_limit(headers)
    assert info is not None
    assert info.requests_remaining == 99
    assert info.requests_limit == 100
    assert info.tokens_remaining == 9000
    assert info.tokens_limit == 10000
    assert info.reset_at == 1700000000.0


def test_extract_rate_limit_partial() -> None:
    headers = {"x-ratelimit-remaining-requests": "5"}
    info = extract_rate_limit(headers)
    assert info is not None
    assert info.requests_remaining == 5
    assert info.requests_limit is None


def test_extract_rate_limit_none_when_no_headers() -> None:
    info = extract_rate_limit({})
    assert info is None


def test_extract_rate_limit_none_for_unrelated_headers() -> None:
    info = extract_rate_limit({"content-type": "application/json"})
    assert info is None


# ---------------------------------------------------------------------------
# classify_error_message
# ---------------------------------------------------------------------------


def test_classify_not_found() -> None:
    assert classify_error_message("Model not found") == "not_found"
    assert classify_error_message("Resource does not exist") == "not_found"


def test_classify_auth() -> None:
    assert classify_error_message("Unauthorized access") == "auth"
    assert classify_error_message("Invalid key provided") == "auth"


def test_classify_context_length() -> None:
    assert classify_error_message("context length exceeded") == "context_length"
    assert classify_error_message("Too many tokens in request") == "context_length"


def test_classify_content_filter() -> None:
    assert classify_error_message("blocked by content filter") == "content_filter"
    assert classify_error_message("Safety violation detected") == "content_filter"


def test_classify_no_match() -> None:
    assert classify_error_message("some random error") is None


def test_classify_case_insensitive() -> None:
    assert classify_error_message("NOT FOUND") == "not_found"
    assert classify_error_message("UNAUTHORIZED") == "auth"
