"""Tests for unified_llm.middleware."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Callable

import pytest

from unified_llm.middleware import (
    CostTracker,
    Middleware,
    cost_tracking_middleware,
    logging_middleware,
)
from unified_llm.types.request import Request
from unified_llm.types.response import Response, Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(model: str = "test-model", provider: str | None = None) -> Request:
    return Request(model=model, provider=provider)


def _make_response(
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
    **kwargs,
) -> Response:
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens
    return Response(
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
        ),
        **kwargs,
    )


def _make_next_fn(response: Response) -> Callable[[Request], Response]:
    """Create a simple next_fn that returns a fixed response."""
    calls: list[Request] = []

    def next_fn(req: Request) -> Response:
        calls.append(req)
        return response

    next_fn.calls = calls  # type: ignore[attr-defined]
    return next_fn


# ---------------------------------------------------------------------------
# TestLoggingMiddleware
# ---------------------------------------------------------------------------


class TestLoggingMiddleware:
    def test_logs_request_info(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = _make_response(total_tokens=42)
        mw = logging_middleware()
        next_fn = _make_next_fn(resp)
        req = _make_request(model="gpt-4", provider="openai")

        with caplog.at_level(logging.INFO, logger="unified_llm"):
            mw(req, next_fn)

        assert any("LLM request" in r.message for r in caplog.records)
        assert any("provider=openai" in r.message for r in caplog.records)
        assert any("model=gpt-4" in r.message for r in caplog.records)

    def test_logs_response_info(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = _make_response(total_tokens=100)
        mw = logging_middleware()
        next_fn = _make_next_fn(resp)

        with caplog.at_level(logging.INFO, logger="unified_llm"):
            mw(_make_request(), next_fn)

        assert any("LLM response" in r.message for r in caplog.records)
        assert any("tokens=100" in r.message for r in caplog.records)

    def test_logs_latency(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = _make_response(total_tokens=0)
        mw = logging_middleware()
        next_fn = _make_next_fn(resp)

        with caplog.at_level(logging.INFO, logger="unified_llm"):
            mw(_make_request(), next_fn)

        response_records = [r for r in caplog.records if "latency=" in r.message]
        assert len(response_records) == 1

    def test_uses_custom_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        custom_logger = logging.getLogger("my_custom_logger")
        resp = _make_response(total_tokens=0)
        mw = logging_middleware(logger=custom_logger)
        next_fn = _make_next_fn(resp)

        with caplog.at_level(logging.INFO, logger="my_custom_logger"):
            mw(_make_request(), next_fn)

        assert any(r.name == "my_custom_logger" for r in caplog.records)

    def test_default_provider_label(self, caplog: pytest.LogCaptureFixture) -> None:
        """When request.provider is None, logs 'default'."""
        resp = _make_response(total_tokens=0)
        mw = logging_middleware()
        next_fn = _make_next_fn(resp)
        req = _make_request(provider=None)

        with caplog.at_level(logging.INFO, logger="unified_llm"):
            mw(req, next_fn)

        assert any("provider=default" in r.message for r in caplog.records)

    def test_returns_response(self) -> None:
        resp = _make_response(model="returned")
        mw = logging_middleware()
        next_fn = _make_next_fn(resp)
        result = mw(_make_request(), next_fn)
        assert result is resp

    def test_calls_next_fn(self) -> None:
        resp = _make_response()
        next_fn = _make_next_fn(resp)
        mw = logging_middleware()
        mw(_make_request(model="special"), next_fn)
        assert len(next_fn.calls) == 1  # type: ignore[attr-defined]
        assert next_fn.calls[0].model == "special"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestCostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_defaults(self) -> None:
        t = CostTracker()
        assert t.total_input_tokens == 0
        assert t.total_output_tokens == 0
        assert t.total_cost == 0.0
        assert t.requests == 0

    def test_manual_mutation(self) -> None:
        t = CostTracker()
        t.total_input_tokens = 100
        t.total_output_tokens = 200
        t.total_cost = 0.50
        t.requests = 3
        assert t.total_input_tokens == 100
        assert t.total_output_tokens == 200
        assert t.total_cost == 0.50
        assert t.requests == 3


# ---------------------------------------------------------------------------
# TestCostTrackingMiddleware
# ---------------------------------------------------------------------------


class TestCostTrackingMiddleware:
    def test_updates_tracker_single_call(self) -> None:
        tracker = CostTracker()
        resp = _make_response(input_tokens=10, output_tokens=20)
        mw = cost_tracking_middleware(tracker)
        next_fn = _make_next_fn(resp)
        mw(_make_request(), next_fn)
        assert tracker.total_input_tokens == 10
        assert tracker.total_output_tokens == 20
        assert tracker.requests == 1

    def test_accumulates_over_multiple_calls(self) -> None:
        tracker = CostTracker()
        mw = cost_tracking_middleware(tracker)

        resp1 = _make_response(input_tokens=10, output_tokens=5)
        resp2 = _make_response(input_tokens=20, output_tokens=15)

        mw(_make_request(), _make_next_fn(resp1))
        mw(_make_request(), _make_next_fn(resp2))

        assert tracker.total_input_tokens == 30
        assert tracker.total_output_tokens == 20
        assert tracker.requests == 2

    def test_returns_response_unmodified(self) -> None:
        tracker = CostTracker()
        resp = _make_response(input_tokens=5, output_tokens=3, model="gpt-4")
        mw = cost_tracking_middleware(tracker)
        result = mw(_make_request(), _make_next_fn(resp))
        assert result is resp

    def test_calls_next_fn(self) -> None:
        tracker = CostTracker()
        resp = _make_response()
        next_fn = _make_next_fn(resp)
        mw = cost_tracking_middleware(tracker)
        mw(_make_request(), next_fn)
        assert len(next_fn.calls) == 1  # type: ignore[attr-defined]

    def test_zero_token_response(self) -> None:
        tracker = CostTracker()
        resp = _make_response(input_tokens=0, output_tokens=0)
        mw = cost_tracking_middleware(tracker)
        mw(_make_request(), _make_next_fn(resp))
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.requests == 1
