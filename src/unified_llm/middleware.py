"""Built-in middleware."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from unified_llm.types.request import Request
from unified_llm.types.response import Response

Middleware = Callable[[Request, Callable[[Request], Response]], Response]


def logging_middleware(logger: logging.Logger | None = None) -> Middleware:
    """Create middleware that logs request/response details."""
    log = logger or logging.getLogger("unified_llm")

    def middleware(
        request: Request, next_fn: Callable[[Request], Response]
    ) -> Response:
        provider = request.provider or "default"
        log.info("LLM request: provider=%s model=%s", provider, request.model)
        start = time.monotonic()
        response = next_fn(request)
        elapsed = time.monotonic() - start
        log.info(
            "LLM response: tokens=%d latency=%.2fs",
            response.usage.total_tokens,
            elapsed,
        )
        return response

    return middleware


@dataclass
class CostTracker:
    """Tracks cumulative token usage and estimated cost."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    requests: int = 0


def cost_tracking_middleware(tracker: CostTracker) -> Middleware:
    """Create middleware that tracks token usage on a CostTracker."""

    def middleware(
        request: Request, next_fn: Callable[[Request], Response]
    ) -> Response:
        response = next_fn(request)
        tracker.total_input_tokens += response.usage.input_tokens
        tracker.total_output_tokens += response.usage.output_tokens
        tracker.requests += 1
        return response

    return middleware
