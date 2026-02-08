"""Provider adapter interface."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from unified_llm.types.request import Request
from unified_llm.types.response import Response
from unified_llm.types.streaming import StreamEvent


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol that every provider adapter must satisfy."""

    @property
    def name(self) -> str:
        """Unique provider name."""
        ...

    def complete(self, request: Request) -> Response:
        """Send a request and return the full response."""
        ...

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """Send a request and yield streaming events."""
        ...


class BaseAdapter:
    """Concrete base class with sensible defaults for provider adapters."""

    @property
    def name(self) -> str:
        """Unique provider name. Subclasses must override."""
        raise NotImplementedError

    def complete(self, request: Request) -> Response:
        """Send a request and return the full response. Subclasses must override."""
        raise NotImplementedError

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        """Send a request and yield streaming events. Subclasses must override."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. Override if the adapter holds connections."""

    def initialize(self) -> None:
        """Perform deferred initialisation. Override if needed."""

    def supports_tool_choice(self, mode: str) -> bool:
        """Return whether the adapter supports a given tool-choice mode."""
        return True


class StubAdapter:
    """In-memory adapter for testing."""

    def __init__(
        self,
        name: str = "stub",
        responses: list[Response] | None = None,
        stream_events: list[list[StreamEvent]] | None = None,
    ) -> None:
        self._name = name
        self._responses = responses or []
        self._stream_events = stream_events or []
        self._response_idx = 0
        self._stream_idx = 0
        self.requests: list[Request] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, request: Request) -> Response:
        self.requests.append(request)
        if not self._responses:
            return Response()
        idx = self._response_idx % len(self._responses)
        self._response_idx += 1
        return self._responses[idx]

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        self.requests.append(request)
        if not self._stream_events:
            return
        idx = self._stream_idx % len(self._stream_events)
        self._stream_idx += 1
        yield from self._stream_events[idx]
