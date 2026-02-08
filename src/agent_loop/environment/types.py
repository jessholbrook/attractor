"""Execution environment types and protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ExecResult:
    """Result of executing a shell command."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_ms: int = 0


@dataclass(frozen=True)
class DirEntry:
    """A single entry from a directory listing."""

    name: str
    is_dir: bool
    size: int | None = None


@dataclass(frozen=True)
class GrepOptions:
    """Options for grep search."""

    case_insensitive: bool = False
    glob_filter: str | None = None
    max_results: int = 100


class ExecutionEnvironment(Protocol):
    """Protocol for where tools run.

    The default is local. Implementations can target Docker, Kubernetes,
    WASM, SSH, or any remote host without changing tool logic.
    """

    # File operations
    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    def file_exists(self, path: str) -> bool: ...
    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]: ...

    # Command execution
    def exec_command(
        self,
        command: str,
        timeout_ms: int = 10_000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult: ...

    # Search operations
    def grep(self, pattern: str, path: str = ".", options: GrepOptions | None = None) -> str: ...
    def glob(self, pattern: str, path: str = ".") -> list[str]: ...

    # Lifecycle
    def initialize(self) -> None: ...
    def cleanup(self) -> None: ...

    # Metadata
    @property
    def working_directory(self) -> str: ...
    @property
    def platform(self) -> str: ...
    @property
    def os_version(self) -> str: ...
