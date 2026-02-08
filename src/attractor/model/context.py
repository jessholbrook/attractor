"""Thread-safe key-value context store for pipeline execution."""

from __future__ import annotations

import copy
import threading
from typing import Any


class Context:
    """Thread-safe key-value store carrying state through a pipeline run.

    All public methods are protected by a threading lock so the context
    can be shared safely across concurrent node executions.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(initial) if initial else {}
        self._log: list[str] = []

    # --- read / write ---------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Set a single key to the given value."""
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key, returning *default* if absent."""
        with self._lock:
            return self._data.get(key, default)

    def get_string(self, key: str, default: str = "") -> str:
        """Retrieve a value as a string, returning *default* if absent."""
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return default
            return str(value)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    # --- bulk operations ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the current data dictionary."""
        with self._lock:
            return dict(self._data)

    def clone(self) -> Context:
        """Return a deep copy suitable for parallel branch isolation."""
        with self._lock:
            ctx = Context.__new__(Context)
            ctx._lock = threading.Lock()
            ctx._data = copy.deepcopy(self._data)
            ctx._log = list(self._log)
            return ctx

    def apply_updates(self, updates: dict[str, Any]) -> None:
        """Merge a dictionary of updates into the context."""
        with self._lock:
            self._data.update(updates)

    # --- logging --------------------------------------------------------------

    def append_log(self, entry: str) -> None:
        """Append a timestamped-or-plain log entry."""
        with self._lock:
            self._log.append(entry)

    @property
    def logs(self) -> list[str]:
        """Return a copy of all log entries."""
        with self._lock:
            return list(self._log)

    # --- dunder helpers -------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            return f"Context(keys={list(self._data.keys())}, log_entries={len(self._log)})"
