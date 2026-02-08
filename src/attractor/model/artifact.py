"""Artifact store: in-memory with automatic file-backing for large payloads."""

from __future__ import annotations

import json
import pickle
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Artifacts larger than this threshold (in bytes) are spilled to disk.
_SPILL_THRESHOLD = 100 * 1024  # 100 KB


@dataclass(frozen=True)
class ArtifactInfo:
    """Metadata about a stored artifact."""

    artifact_id: str
    name: str
    size_bytes: int
    spilled: bool


@dataclass
class ArtifactStore:
    """Storage backend for pipeline artifacts.

    Small artifacts (<=100 KB serialised) are kept in memory. Larger ones
    are automatically spilled to a temporary directory on disk and loaded
    on demand.
    """

    _spill_dir: Path | None = field(default=None, repr=False)
    _memory: dict[str, Any] = field(default_factory=dict, repr=False)
    _info: dict[str, ArtifactInfo] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- public API -----------------------------------------------------------

    def store(self, artifact_id: str, name: str, data: Any) -> ArtifactInfo:
        """Store an artifact, spilling to disk if the payload exceeds 100 KB.

        Returns an ``ArtifactInfo`` describing what was stored.
        """
        serialised = self._serialise(data)
        size = len(serialised)
        spilled = size > _SPILL_THRESHOLD

        with self._lock:
            if spilled:
                path = self._spill_path(artifact_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(serialised)
                self._memory.pop(artifact_id, None)
            else:
                self._memory[artifact_id] = data
                # Clean up any prior spill file
                self._remove_spill(artifact_id)

            info = ArtifactInfo(
                artifact_id=artifact_id,
                name=name,
                size_bytes=size,
                spilled=spilled,
            )
            self._info[artifact_id] = info
            return info

    def retrieve(self, artifact_id: str) -> Any:
        """Retrieve a previously stored artifact by its ID.

        Raises ``KeyError`` if the artifact does not exist.
        """
        with self._lock:
            if artifact_id not in self._info:
                raise KeyError(f"No artifact with id {artifact_id!r}")

            info = self._info[artifact_id]

            if not info.spilled:
                return self._memory[artifact_id]

            path = self._spill_path(artifact_id)
            return self._deserialise(path.read_bytes())

    def has(self, artifact_id: str) -> bool:
        """Check whether an artifact exists in the store."""
        with self._lock:
            return artifact_id in self._info

    def list_artifacts(self) -> list[ArtifactInfo]:
        """Return metadata for all stored artifacts."""
        with self._lock:
            return list(self._info.values())

    # --- internal helpers -----------------------------------------------------

    def _ensure_spill_dir(self) -> Path:
        if self._spill_dir is None:
            self._spill_dir = Path(tempfile.mkdtemp(prefix="attractor_artifacts_"))
        return self._spill_dir

    def _spill_path(self, artifact_id: str) -> Path:
        # Use a safe filename derived from the artifact id
        safe_name = artifact_id.replace("/", "_").replace("\\", "_")
        return self._ensure_spill_dir() / f"{safe_name}.bin"

    def _remove_spill(self, artifact_id: str) -> None:
        path = self._spill_path(artifact_id)
        if path.exists():
            path.unlink()

    @staticmethod
    def _serialise(data: Any) -> bytes:
        """Serialise data to bytes.

        Tries JSON first (for portability); falls back to pickle for
        arbitrary Python objects.
        """
        try:
            return json.dumps(data, default=str).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            return pickle.dumps(data)

    @staticmethod
    def _deserialise(raw: bytes) -> Any:
        """Deserialise bytes produced by ``_serialise``."""
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return pickle.loads(raw)  # noqa: S301
