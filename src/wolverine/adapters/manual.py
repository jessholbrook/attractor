from __future__ import annotations

import uuid
from datetime import datetime, timezone

from wolverine.model.signal import RawSignal, SignalKind, SignalSource


class ManualAdapter:
    """In-memory adapter for manually submitted signals.

    Use ``submit()`` to enqueue signals, then ``fetch()`` to drain them.
    Source is always ``SignalSource.CLI``.
    """

    def __init__(self) -> None:
        self._pending: list[RawSignal] = []

    @property
    def source(self) -> SignalSource:
        return SignalSource.CLI

    def submit(
        self,
        title: str,
        body: str,
        kind: SignalKind = SignalKind.MANUAL,
    ) -> RawSignal:
        """Add a signal to the pending queue and return it."""
        signal = RawSignal(
            id=uuid.uuid4().hex,
            kind=kind,
            source=SignalSource.CLI,
            title=title,
            body=body,
            received_at=datetime.now(timezone.utc).isoformat(),
        )
        self._pending.append(signal)
        return signal

    def fetch(self) -> tuple[RawSignal, ...]:
        """Drain and return all pending signals."""
        signals = tuple(self._pending)
        self._pending.clear()
        return signals
