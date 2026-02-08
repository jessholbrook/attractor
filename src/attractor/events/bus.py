"""Simple synchronous event bus for pipeline lifecycle events."""

from typing import Any, Callable


class EventBus:
    """Synchronous publish-subscribe event bus.

    Listeners can subscribe to specific event types or receive all events.
    Events are dispatched synchronously in registration order.
    """

    def __init__(self) -> None:
        self._listeners: dict[type, list[Callable]] = {}
        self._global_listeners: list[Callable] = []

    def subscribe(self, event_type: type, callback: Callable) -> None:
        """Register a callback for a specific event type."""
        self._listeners.setdefault(event_type, []).append(callback)

    def on_all(self, callback: Callable) -> None:
        """Register a callback that receives every event."""
        self._global_listeners.append(callback)

    def emit(self, event: Any) -> None:
        """Dispatch an event to all matching listeners."""
        for cb in self._global_listeners:
            cb(event)
        for cb in self._listeners.get(type(event), []):
            cb(event)
