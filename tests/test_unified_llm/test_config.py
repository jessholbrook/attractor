"""Tests for unified_llm.types.config."""
from __future__ import annotations

import pytest

from unified_llm.types.config import (
    AbortController,
    AbortSignal,
    AdapterTimeout,
    RetryPolicy,
    TimeoutConfig,
)


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self) -> None:
        rp = RetryPolicy()
        assert rp.max_retries == 2
        assert rp.base_delay == 1.0
        assert rp.max_delay == 60.0
        assert rp.backoff_multiplier == 2.0
        assert rp.jitter is True
        assert rp.on_retry is None

    def test_custom_values(self) -> None:
        rp = RetryPolicy(max_retries=5, base_delay=0.5, max_delay=30.0)
        assert rp.max_retries == 5
        assert rp.base_delay == 0.5
        assert rp.max_delay == 30.0

    def test_frozen(self) -> None:
        rp = RetryPolicy()
        with pytest.raises(AttributeError):
            rp.max_retries = 10  # type: ignore[misc]

    def test_on_retry_callable(self) -> None:
        calls: list[tuple[int, Exception, float]] = []

        def handler(attempt: int, exc: Exception, delay: float) -> None:
            calls.append((attempt, exc, delay))

        rp = RetryPolicy(on_retry=handler)
        assert rp.on_retry is handler

    def test_on_retry_excluded_from_comparison(self) -> None:
        def handler_a(a: int, e: Exception, d: float) -> None:
            pass

        def handler_b(a: int, e: Exception, d: float) -> None:
            pass

        rp_a = RetryPolicy(on_retry=handler_a)
        rp_b = RetryPolicy(on_retry=handler_b)
        # on_retry is compare=False, so these should be equal
        assert rp_a == rp_b

    def test_equality(self) -> None:
        assert RetryPolicy() == RetryPolicy()
        assert RetryPolicy(max_retries=3) != RetryPolicy(max_retries=2)


# ---------------------------------------------------------------------------
# TimeoutConfig
# ---------------------------------------------------------------------------


class TestTimeoutConfig:
    def test_defaults(self) -> None:
        tc = TimeoutConfig()
        assert tc.total is None
        assert tc.per_step is None

    def test_custom(self) -> None:
        tc = TimeoutConfig(total=120.0, per_step=30.0)
        assert tc.total == 120.0
        assert tc.per_step == 30.0

    def test_frozen(self) -> None:
        tc = TimeoutConfig()
        with pytest.raises(AttributeError):
            tc.total = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AdapterTimeout
# ---------------------------------------------------------------------------


class TestAdapterTimeout:
    def test_defaults(self) -> None:
        at = AdapterTimeout()
        assert at.connect == 5.0
        assert at.request == 60.0
        assert at.stream_read == 30.0

    def test_custom(self) -> None:
        at = AdapterTimeout(connect=10.0, request=120.0, stream_read=60.0)
        assert at.connect == 10.0
        assert at.request == 120.0
        assert at.stream_read == 60.0

    def test_frozen(self) -> None:
        at = AdapterTimeout()
        with pytest.raises(AttributeError):
            at.connect = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AbortSignal
# ---------------------------------------------------------------------------


class TestAbortSignal:
    def test_initial_state(self) -> None:
        sig = AbortSignal()
        assert sig.aborted is False

    def test_abort(self) -> None:
        sig = AbortSignal()
        sig._abort()
        assert sig.aborted is True

    def test_multiple_abort_calls(self) -> None:
        sig = AbortSignal()
        sig._abort()
        sig._abort()
        assert sig.aborted is True


# ---------------------------------------------------------------------------
# AbortController
# ---------------------------------------------------------------------------


class TestAbortController:
    def test_initial_signal(self) -> None:
        ctrl = AbortController()
        assert isinstance(ctrl.signal, AbortSignal)
        assert ctrl.signal.aborted is False

    def test_abort_sets_signal(self) -> None:
        ctrl = AbortController()
        ctrl.abort()
        assert ctrl.signal.aborted is True

    def test_separate_controllers_independent(self) -> None:
        ctrl_a = AbortController()
        ctrl_b = AbortController()
        ctrl_a.abort()
        assert ctrl_a.signal.aborted is True
        assert ctrl_b.signal.aborted is False
