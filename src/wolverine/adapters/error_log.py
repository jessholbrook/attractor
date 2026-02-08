from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from wolverine.model.signal import RawSignal, SignalKind, SignalSource


class ErrorLogAdapter:
    """Adapter that watches a directory for .log and .json files.

    Each file becomes a signal (title = filename, body = file content).
    Tracks processed filenames to avoid re-ingestion on subsequent fetch calls.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._processed: set[str] = set()

    @property
    def source(self) -> SignalSource:
        return SignalSource.CLI

    def fetch(self) -> tuple[RawSignal, ...]:
        """Scan directory for new .log and .json files."""
        if not self._directory.is_dir():
            return ()

        signals: list[RawSignal] = []
        now = datetime.now(timezone.utc).isoformat()

        for pattern in ("*.log", "*.json"):
            for filepath in sorted(self._directory.glob(pattern)):
                name = filepath.name
                if name in self._processed:
                    continue

                try:
                    body = filepath.read_text(encoding="utf-8")
                except OSError:
                    continue

                signal = RawSignal(
                    id=uuid.uuid4().hex,
                    kind=SignalKind.ERROR_LOG,
                    source=SignalSource.CLI,
                    title=name,
                    body=body,
                    received_at=now,
                )
                signals.append(signal)
                self._processed.add(name)

        return tuple(signals)
