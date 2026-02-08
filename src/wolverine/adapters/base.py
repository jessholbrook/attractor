from __future__ import annotations

from typing import Protocol

from wolverine.model.signal import RawSignal, SignalSource


class SignalAdapter(Protocol):
    """Protocol for signal source adapters."""

    def fetch(self) -> tuple[RawSignal, ...]:
        """Fetch new signals. Returns signals not yet ingested."""
        ...

    @property
    def source(self) -> SignalSource: ...
