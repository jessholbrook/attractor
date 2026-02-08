"""Tests for retry logic: policies, backoff, and policy building."""

import pytest

from attractor.engine.retry import BackoffConfig, RetryPolicy, build_retry_policy
from attractor.model.graph import Graph, Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(max_retries: int = 0, **kwargs) -> Node:
    return Node(id="test_node", max_retries=max_retries, **kwargs)


def _graph(attributes: dict[str, str] | None = None) -> Graph:
    return Graph(name="test", attributes=attributes or {})


# ---------------------------------------------------------------------------
# RetryPolicy basics
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    def test_max_attempts_one_means_no_retries(self):
        policy = RetryPolicy(max_attempts=1, backoff=BackoffConfig())
        assert policy.max_attempts == 1

    def test_delay_for_attempt_grows_exponentially(self):
        backoff = BackoffConfig(initial_delay_ms=100, backoff_factor=2.0, jitter=False)
        policy = RetryPolicy(max_attempts=5, backoff=backoff)
        # Attempt 1: 100ms = 0.1s
        assert policy.delay_for_attempt(1) == pytest.approx(0.1)
        # Attempt 2: 200ms = 0.2s
        assert policy.delay_for_attempt(2) == pytest.approx(0.2)
        # Attempt 3: 400ms = 0.4s
        assert policy.delay_for_attempt(3) == pytest.approx(0.4)
        # Attempt 4: 800ms = 0.8s
        assert policy.delay_for_attempt(4) == pytest.approx(0.8)

    def test_delay_capped_at_max(self):
        backoff = BackoffConfig(
            initial_delay_ms=1000, backoff_factor=10.0, max_delay_ms=5000, jitter=False
        )
        policy = RetryPolicy(max_attempts=5, backoff=backoff)
        # Attempt 1: 1000ms
        assert policy.delay_for_attempt(1) == pytest.approx(1.0)
        # Attempt 2: 10000ms -> capped at 5000ms = 5.0s
        assert policy.delay_for_attempt(2) == pytest.approx(5.0)
        # Attempt 3: 100000ms -> capped at 5000ms = 5.0s
        assert policy.delay_for_attempt(3) == pytest.approx(5.0)

    def test_jitter_varies_delay(self):
        backoff = BackoffConfig(initial_delay_ms=1000, backoff_factor=1.0, jitter=True)
        policy = RetryPolicy(max_attempts=5, backoff=backoff)
        # With jitter, delay should be between 0.5 and 1.5 seconds for attempt 1
        delays = [policy.delay_for_attempt(1) for _ in range(50)]
        assert min(delays) >= 0.5
        assert max(delays) <= 1.5
        # Not all the same (jitter adds randomness)
        assert len(set(delays)) > 1

    def test_no_jitter_deterministic(self):
        backoff = BackoffConfig(initial_delay_ms=200, backoff_factor=2.0, jitter=False)
        policy = RetryPolicy(max_attempts=3, backoff=backoff)
        d1 = policy.delay_for_attempt(1)
        d2 = policy.delay_for_attempt(1)
        assert d1 == d2

    def test_linear_backoff(self):
        backoff = BackoffConfig(initial_delay_ms=500, backoff_factor=1.0, jitter=False)
        policy = RetryPolicy(max_attempts=3, backoff=backoff)
        assert policy.delay_for_attempt(1) == pytest.approx(0.5)
        assert policy.delay_for_attempt(2) == pytest.approx(0.5)
        assert policy.delay_for_attempt(3) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# build_retry_policy
# ---------------------------------------------------------------------------

class TestBuildRetryPolicy:
    def test_reads_node_max_retries(self):
        node = _node(max_retries=3)
        policy = build_retry_policy(node, _graph())
        assert policy.max_attempts == 4  # 3 retries + 1 initial

    def test_zero_retries_means_single_attempt(self):
        node = _node(max_retries=0)
        policy = build_retry_policy(node, _graph())
        assert policy.max_attempts == 1

    def test_falls_back_to_graph_default(self):
        node = _node(max_retries=0)
        graph = _graph(attributes={"default_max_retry": "2"})
        policy = build_retry_policy(node, graph)
        assert policy.max_attempts == 3  # 2 retries + 1 initial

    def test_node_retries_override_graph_default(self):
        node = _node(max_retries=5)
        graph = _graph(attributes={"default_max_retry": "2"})
        policy = build_retry_policy(node, graph)
        assert policy.max_attempts == 6  # node value takes precedence

    def test_invalid_graph_default_treated_as_zero(self):
        node = _node(max_retries=0)
        graph = _graph(attributes={"default_max_retry": "not_a_number"})
        policy = build_retry_policy(node, graph)
        assert policy.max_attempts == 1
