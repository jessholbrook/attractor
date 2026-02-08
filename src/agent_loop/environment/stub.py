"""Stub execution environment for testing."""

from __future__ import annotations

from agent_loop.environment.types import DirEntry, ExecResult, GrepOptions


class StubExecutionEnvironment:
    """Test stub that returns predefined values for all operations.

    Provides configurable files, exec results, and metadata for unit-testing
    Session and other modules without touching the real filesystem.
    """

    def __init__(
        self,
        working_dir: str = "/stub/workspace",
        plat: str = "darwin",
        os_ver: str = "Darwin 24.0.0",
        files: dict[str, str] | None = None,
        exec_results: list[ExecResult] | None = None,
    ) -> None:
        self._working_dir = working_dir
        self._platform = plat
        self._os_version = os_ver
        self._files: dict[str, str] = dict(files) if files else {}
        self._exec_results = list(exec_results) if exec_results else [ExecResult()]
        self._exec_index = 0
        self._exec_calls: list[str] = []

    # --- File operations ---

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        if path not in self._files:
            raise FileNotFoundError(f"Stub file not found: {path}")
        content = self._files[path]
        lines = content.splitlines(keepends=True)
        start = (offset - 1) if offset and offset > 0 else 0
        end = (start + limit) if limit else None
        return "".join(lines[start:end])

    def write_file(self, path: str, content: str) -> None:
        self._files[path] = content

    def file_exists(self, path: str) -> bool:
        return path in self._files

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        return []

    # --- Command execution ---

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 10_000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        self._exec_calls.append(command)
        if self._exec_index < len(self._exec_results):
            result = self._exec_results[self._exec_index]
            self._exec_index += 1
        else:
            result = self._exec_results[-1]
        return result

    # --- Search operations ---

    def grep(self, pattern: str, path: str = ".", options: GrepOptions | None = None) -> str:
        return ""

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        return [p for p in sorted(self._files.keys()) if p.startswith(path.rstrip("/"))]

    # --- Lifecycle ---

    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    # --- Metadata ---

    @property
    def working_directory(self) -> str:
        return self._working_dir

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def os_version(self) -> str:
        return self._os_version

    # --- Test helpers ---

    @property
    def exec_calls(self) -> list[str]:
        """All commands passed to exec_command, in order."""
        return list(self._exec_calls)
