"""Configuration types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for automatic retry behaviour."""

    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    on_retry: Callable[[int, Exception, float], None] | None = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True)
class TimeoutConfig:
    """High-level timeout configuration."""

    total: float | None = None
    per_step: float | None = None


@dataclass(frozen=True)
class AdapterTimeout:
    """Low-level timeout settings used by adapters."""

    connect: float = 5.0
    request: float = 60.0
    stream_read: float = 30.0


class AbortSignal:
    """An observable flag indicating whether an operation has been aborted."""

    def __init__(self) -> None:
        self._aborted = False

    @property
    def aborted(self) -> bool:
        return self._aborted

    def _abort(self) -> None:
        self._aborted = True


class AbortController:
    """Controls an :class:`AbortSignal` to cancel an in-flight operation."""

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self) -> None:
        self.signal._abort()
