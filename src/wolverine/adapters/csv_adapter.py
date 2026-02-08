from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from wolverine.model.signal import RawSignal, SignalKind, SignalSource


class CSVAdapter:
    """Adapter that reads signals from a CSV file.

    Expected columns: title, body, kind (optional), metadata (optional JSON).
    ``fetch()`` reads the file once, then marks as consumed so subsequent
    calls return an empty tuple.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._consumed = False

    @property
    def source(self) -> SignalSource:
        return SignalSource.CSV

    def fetch(self) -> tuple[RawSignal, ...]:
        """Read all rows from the CSV and return as signals.

        Raises FileNotFoundError if the file does not exist.
        Returns an empty tuple on subsequent calls.
        """
        if self._consumed:
            return ()

        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        signals: list[RawSignal] = []
        now = datetime.now(timezone.utc).isoformat()

        with self._path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                title = row.get("title", "").strip()
                body = row.get("body", "").strip()
                if not title and not body:
                    continue

                kind_str = row.get("kind", "manual").strip() or "manual"
                try:
                    kind = SignalKind(kind_str)
                except ValueError:
                    kind = SignalKind.MANUAL

                metadata: dict[str, str] = {}
                meta_raw = row.get("metadata", "").strip()
                if meta_raw:
                    try:
                        metadata = json.loads(meta_raw)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                signal = RawSignal(
                    id=uuid.uuid4().hex,
                    kind=kind,
                    source=SignalSource.CSV,
                    title=title,
                    body=body,
                    received_at=now,
                    metadata=metadata,
                )
                signals.append(signal)

        self._consumed = True
        return tuple(signals)
