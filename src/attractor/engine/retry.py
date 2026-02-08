"""Retry logic: backoff configuration and retry policy for node execution."""

from __future__ import annotations

import random
from dataclasses import dataclass

from attractor.model.graph import Graph, Node


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff between retries."""

    initial_delay_ms: int = 200
    backoff_factor: float = 2.0
    max_delay_ms: int = 60000
    jitter: bool = True


@dataclass
class RetryPolicy:
    """Retry policy: max_attempts=1 means no retries (single attempt)."""

    max_attempts: int  # 1 = no retries
    backoff: BackoffConfig

    def delay_for_attempt(self, attempt: int) -> float:
        """Return delay in seconds for the given attempt (1-indexed).

        Attempt 1 uses initial_delay_ms, attempt 2 uses initial * factor, etc.
        The delay is capped at max_delay_ms and optionally jittered.
        """
        delay = self.backoff.initial_delay_ms * (self.backoff.backoff_factor ** (attempt - 1))
        delay = min(delay, self.backoff.max_delay_ms)
        if self.backoff.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay / 1000.0  # convert to seconds


# Preset policies
PRESET_POLICIES = {
    "none": RetryPolicy(max_attempts=1, backoff=BackoffConfig()),
    "standard": RetryPolicy(
        max_attempts=5, backoff=BackoffConfig(initial_delay_ms=200, backoff_factor=2.0)
    ),
    "aggressive": RetryPolicy(
        max_attempts=5, backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=2.0)
    ),
    "linear": RetryPolicy(
        max_attempts=3, backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=1.0)
    ),
    "patient": RetryPolicy(
        max_attempts=3, backoff=BackoffConfig(initial_delay_ms=2000, backoff_factor=3.0)
    ),
}


def build_retry_policy(node: Node, graph: Graph) -> RetryPolicy:
    """Build retry policy from node and graph attributes.

    Uses node.max_retries if set, otherwise falls back to the graph-level
    default_max_retry attribute. max_retries is extra attempts beyond the first.
    """
    max_retries = node.max_retries
    if max_retries == 0:
        # Check graph default
        default = graph.attributes.get("default_max_retry", "0")
        try:
            max_retries = int(default)
        except (ValueError, TypeError):
            max_retries = 0
    max_attempts = max_retries + 1  # max_retries is additional attempts
    return RetryPolicy(max_attempts=max_attempts, backoff=BackoffConfig())
