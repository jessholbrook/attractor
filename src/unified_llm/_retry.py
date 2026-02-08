"""Retry logic with exponential backoff."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from unified_llm.errors import SDKError
from unified_llm.types.config import RetryPolicy

T = TypeVar("T")


def calculate_delay(attempt: int, policy: RetryPolicy) -> float:
    """Compute the delay for a given retry attempt.

    Uses exponential backoff clamped to *policy.max_delay*, with optional
    jitter.
    """
    delay = min(
        policy.base_delay * (policy.backoff_multiplier ** attempt),
        policy.max_delay,
    )
    if policy.jitter:
        delay *= random.uniform(0.5, 1.5)
    return delay


def with_retry(fn: Callable[[], T], policy: RetryPolicy) -> T:
    """Execute *fn*, retrying according to *policy* on failure."""
    last_exc: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc

            # Check if this is the last attempt
            if attempt >= policy.max_retries:
                raise

            # Check retryable — unknown errors default to retryable
            retryable = getattr(exc, "retryable", True)
            if not retryable:
                raise

            # Check retry_after
            retry_after: float | None = getattr(exc, "retry_after", None)
            if retry_after is not None and retry_after > policy.max_delay:
                raise

            # Determine delay
            if retry_after is not None and retry_after <= policy.max_delay:
                delay = retry_after
            else:
                delay = calculate_delay(attempt, policy)

            # Notify callback
            if policy.on_retry is not None:
                policy.on_retry(attempt, exc, delay)

            time.sleep(delay)

    # Should be unreachable, but satisfy the type checker
    assert last_exc is not None  # noqa: S101
    raise last_exc
