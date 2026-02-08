"""Response normalization helpers."""
from __future__ import annotations

from typing import Any

from unified_llm.types.enums import FinishReason
from unified_llm.types.response import FinishReasonInfo, RateLimitInfo, Usage


# ---------------------------------------------------------------------------
# Finish reason mapping tables
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP: dict[str, dict[str, FinishReason]] = {
    "openai": {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    },
    "anthropic": {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALLS,
    },
    "gemini": {
        "STOP": FinishReason.STOP,
        "MAX_TOKENS": FinishReason.LENGTH,
        "SAFETY": FinishReason.CONTENT_FILTER,
        "RECITATION": FinishReason.CONTENT_FILTER,
    },
}


def map_finish_reason(raw: str, provider: str) -> FinishReasonInfo:
    """Map a provider-specific finish reason string to a :class:`FinishReasonInfo`."""
    provider_map = _FINISH_REASON_MAP.get(provider, {})
    reason = provider_map.get(raw, FinishReason.OTHER)
    return FinishReasonInfo(reason=reason, raw=raw)


# ---------------------------------------------------------------------------
# Rate limit extraction
# ---------------------------------------------------------------------------

def extract_rate_limit(headers: dict[str, str]) -> RateLimitInfo | None:
    """Extract rate-limit information from response headers.

    Returns ``None`` if no rate-limit headers are present.
    """

    def _int(key: str) -> int | None:
        val = headers.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _float(key: str) -> float | None:
        val = headers.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    remaining_requests = _int("x-ratelimit-remaining-requests")
    limit_requests = _int("x-ratelimit-limit-requests")
    remaining_tokens = _int("x-ratelimit-remaining-tokens")
    limit_tokens = _int("x-ratelimit-limit-tokens")
    reset_at = _float("x-ratelimit-reset")

    if all(
        v is None
        for v in (remaining_requests, limit_requests, remaining_tokens, limit_tokens, reset_at)
    ):
        return None

    return RateLimitInfo(
        requests_remaining=remaining_requests,
        requests_limit=limit_requests,
        tokens_remaining=remaining_tokens,
        tokens_limit=limit_tokens,
        reset_at=reset_at,
    )


# ---------------------------------------------------------------------------
# Error message classification
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[list[str], str]] = [
    (["not found", "does not exist"], "not_found"),
    (["unauthorized", "invalid key"], "auth"),
    (["context length", "too many tokens"], "context_length"),
    (["content filter", "safety"], "content_filter"),
]


def classify_error_message(message: str) -> str | None:
    """Classify an error message into a known category.

    Returns ``None`` if the message does not match any known pattern.
    """
    lower = message.lower()
    for patterns, category in _ERROR_PATTERNS:
        for pattern in patterns:
            if pattern in lower:
                return category
    return None
