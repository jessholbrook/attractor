"""Tests for the retry executor."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from unified_llm._retry import calculate_delay, with_retry
from unified_llm.errors import SDKError, AuthenticationError, ServerError, RateLimitError
from unified_llm.types.config import RetryPolicy


# ---------------------------------------------------------------------------
# calculate_delay
# ---------------------------------------------------------------------------


def test_calculate_delay_exponential() -> None:
    policy = RetryPolicy(base_delay=1.0, backoff_multiplier=2.0, max_delay=60.0, jitter=False)
    assert calculate_delay(0, policy) == 1.0
    assert calculate_delay(1, policy) == 2.0
    assert calculate_delay(2, policy) == 4.0
    assert calculate_delay(3, policy) == 8.0


def test_calculate_delay_clamped_to_max() -> None:
    policy = RetryPolicy(base_delay=1.0, backoff_multiplier=10.0, max_delay=5.0, jitter=False)
    assert calculate_delay(0, policy) == 1.0
    assert calculate_delay(1, policy) == 5.0  # 10 clamped to 5
    assert calculate_delay(2, policy) == 5.0


def test_calculate_delay_with_jitter() -> None:
    policy = RetryPolicy(base_delay=10.0, backoff_multiplier=1.0, max_delay=60.0, jitter=True)
    delays = [calculate_delay(0, policy) for _ in range(100)]
    # With jitter factor uniform(0.5, 1.5), range is 5.0 to 15.0
    assert all(4.9 <= d <= 15.1 for d in delays)
    # Should not all be the same
    assert len(set(delays)) > 1


# ---------------------------------------------------------------------------
# with_retry — success
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_success_on_first_call(mock_sleep: MagicMock) -> None:
    result = with_retry(lambda: 42, RetryPolicy())
    assert result == 42
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# with_retry — retryable errors
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_retry_on_retryable_error(mock_sleep: MagicMock) -> None:
    call_count = 0

    def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ServerError("server error")
        return "ok"

    policy = RetryPolicy(max_retries=3, jitter=False)
    result = with_retry(fn, policy)
    assert result == "ok"
    assert call_count == 3
    assert mock_sleep.call_count == 2


@patch("unified_llm._retry.time.sleep")
def test_max_retries_exhausted(mock_sleep: MagicMock) -> None:
    def fn() -> str:
        raise ServerError("always fails")

    policy = RetryPolicy(max_retries=2, jitter=False)
    with pytest.raises(ServerError, match="always fails"):
        with_retry(fn, policy)
    # 2 retries = 2 sleeps
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# with_retry — non-retryable errors
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_non_retryable_error_not_retried(mock_sleep: MagicMock) -> None:
    def fn() -> str:
        raise AuthenticationError("bad key")

    policy = RetryPolicy(max_retries=3)
    with pytest.raises(AuthenticationError, match="bad key"):
        with_retry(fn, policy)
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# with_retry — unknown exceptions default retryable
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_unknown_exception_retried(mock_sleep: MagicMock) -> None:
    call_count = 0

    def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("oops")
        return "recovered"

    policy = RetryPolicy(max_retries=2, jitter=False)
    result = with_retry(fn, policy)
    assert result == "recovered"
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# with_retry — on_retry callback
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_on_retry_callback(mock_sleep: MagicMock) -> None:
    calls: list[tuple[int, Exception, float]] = []

    def on_retry(attempt: int, exc: Exception, delay: float) -> None:
        calls.append((attempt, exc, delay))

    call_count = 0

    def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ServerError("fail")
        return "ok"

    policy = RetryPolicy(max_retries=3, jitter=False, on_retry=on_retry)
    with_retry(fn, policy)
    assert len(calls) == 2
    assert calls[0][0] == 0  # attempt 0
    assert calls[1][0] == 1  # attempt 1


# ---------------------------------------------------------------------------
# with_retry — retry_after
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_retry_after_respected(mock_sleep: MagicMock) -> None:
    call_count = 0

    def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RateLimitError("rate limit", retry_after=5.0)
        return "ok"

    policy = RetryPolicy(max_retries=2, max_delay=60.0, jitter=False)
    result = with_retry(fn, policy)
    assert result == "ok"
    mock_sleep.assert_called_once_with(5.0)


@patch("unified_llm._retry.time.sleep")
def test_retry_after_exceeding_max_delay_raises(mock_sleep: MagicMock) -> None:
    def fn() -> str:
        raise RateLimitError("rate limit", retry_after=120.0)

    policy = RetryPolicy(max_retries=3, max_delay=60.0)
    with pytest.raises(RateLimitError):
        with_retry(fn, policy)
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# with_retry — jitter applied
# ---------------------------------------------------------------------------


@patch("unified_llm._retry.time.sleep")
def test_jitter_applied_to_delay(mock_sleep: MagicMock) -> None:
    call_count = 0

    def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ServerError("fail")
        return "ok"

    policy = RetryPolicy(max_retries=2, base_delay=10.0, jitter=True)
    with_retry(fn, policy)
    assert mock_sleep.call_count == 1
    actual_delay = mock_sleep.call_args[0][0]
    # jitter range: 10 * uniform(0.5, 1.5) -> [5.0, 15.0]
    assert 4.9 <= actual_delay <= 15.1
