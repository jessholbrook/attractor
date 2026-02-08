from __future__ import annotations

from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.store.db import Database


class FeedbackAdapter:
    """Adapter that reads unprocessed signals from the wolverine SQLite database.

    Reads signals with ``source='form'`` that have a ``received_at`` after the
    high-water mark. This prevents re-processing.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._high_water: str = ""

    @property
    def source(self) -> SignalSource:
        return SignalSource.FORM

    def fetch(self) -> tuple[RawSignal, ...]:
        """Fetch signals with source='form' after the high-water mark."""
        import json

        if self._high_water:
            rows = self._db.fetch_all(
                "SELECT * FROM signals WHERE source = ? AND received_at > ? ORDER BY received_at ASC",
                ("form", self._high_water),
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM signals WHERE source = ? ORDER BY received_at ASC",
                ("form",),
            )

        if not rows:
            return ()

        signals: list[RawSignal] = []
        for row in rows:
            signals.append(
                RawSignal(
                    id=row["id"],
                    kind=SignalKind(row["kind"]),
                    source=SignalSource(row["source"]),
                    title=row["title"],
                    body=row["body"],
                    received_at=row["received_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    raw_payload=row["raw_payload"],
                )
            )

        # Update high-water mark to the latest received_at
        self._high_water = signals[-1].received_at
        return tuple(signals)
