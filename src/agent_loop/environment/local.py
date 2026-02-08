"""Local execution environment: runs tools on the local machine."""

from __future__ import annotations

import fnmatch
import os
import platform
import re
import signal
import subprocess
import time
from enum import Enum
from pathlib import Path

from agent_loop.environment.types import DirEntry, ExecResult, GrepOptions


# --- Environment variable filtering ---

SENSITIVE_PATTERNS = [
    "*_API_KEY", "*_SECRET", "*_TOKEN", "*_PASSWORD", "*_CREDENTIAL",
]

ALWAYS_INCLUDE = [
    "PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR",
    "GOPATH", "CARGO_HOME", "NVM_DIR", "PYTHONPATH", "VIRTUAL_ENV",
    "PYENV_ROOT", "RBENV_ROOT", "RUSTUP_HOME",
]


class EnvVarPolicy(Enum):
    INHERIT_CORE = "inherit_core"    # Default: filter sensitive, keep core
    INHERIT_ALL = "inherit_all"      # Pass everything through
    INHERIT_NONE = "inherit_none"    # Only ALWAYS_INCLUDE


def _is_sensitive(name: str) -> bool:
    upper = name.upper()
    return any(fnmatch.fnmatch(upper, pat) for pat in SENSITIVE_PATTERNS)


def _filter_env(policy: EnvVarPolicy, extra: dict[str, str] | None = None) -> dict[str, str]:
    base = os.environ.copy()

    if policy == EnvVarPolicy.INHERIT_ALL:
        env = base
    elif policy == EnvVarPolicy.INHERIT_NONE:
        env = {k: v for k, v in base.items() if k in ALWAYS_INCLUDE}
    else:  # INHERIT_CORE
        env = {k: v for k, v in base.items() if not _is_sensitive(k)}

    if extra:
        env.update(extra)
    return env


class LocalExecutionEnvironment:
    """Runs tools on the local machine.

    File operations use pathlib. Command execution uses subprocess with
    process groups for clean timeout handling. Grep tries ripgrep first,
    falls back to Python re. Glob uses pathlib.
    """

    def __init__(
        self,
        working_dir: str | None = None,
        env_policy: EnvVarPolicy = EnvVarPolicy.INHERIT_CORE,
    ) -> None:
        self._working_dir = working_dir or os.getcwd()
        self._env_policy = env_policy

    # --- File operations ---

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        resolved = self._resolve(path)
        content = resolved.read_text(encoding="utf-8", errors="replace")
        if offset is None and limit is None:
            return content
        lines = content.splitlines(keepends=True)
        start = (offset - 1) if offset and offset > 0 else 0
        end = (start + limit) if limit else None
        return "".join(lines[start:end])

    def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def file_exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        resolved = self._resolve(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        entries = []
        for child in sorted(resolved.iterdir()):
            entries.append(DirEntry(
                name=child.name,
                is_dir=child.is_dir(),
                size=child.stat().st_size if child.is_file() else None,
            ))
        return entries

    # --- Command execution ---

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 10_000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cwd = working_dir or self._working_dir
        env = _filter_env(self._env_policy, env_vars)
        timeout_s = timeout_ms / 1000.0
        start = time.monotonic()

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
            duration = int((time.monotonic() - start) * 1000)
            return ExecResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=proc.returncode,
                timed_out=False,
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired:
            # SIGTERM to process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            # Wait 2 seconds for graceful shutdown
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                # SIGKILL
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                stdout_bytes, stderr_bytes = proc.communicate()

            duration = int((time.monotonic() - start) * 1000)
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            timeout_msg = f"\n[ERROR: Command timed out after {timeout_ms}ms. Partial output is shown above.\nYou can retry with a longer timeout by setting the timeout_ms parameter.]"
            return ExecResult(
                stdout=stdout + timeout_msg,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
                duration_ms=duration,
            )

    # --- Search operations ---

    def grep(self, pattern: str, path: str = ".", options: GrepOptions | None = None) -> str:
        opts = options or GrepOptions()
        search_path = self._resolve(path)

        # Try ripgrep first
        try:
            return self._grep_rg(pattern, str(search_path), opts)
        except FileNotFoundError:
            pass

        # Fallback to Python re
        return self._grep_python(pattern, search_path, opts)

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        base = self._resolve(path)
        matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(m) for m in matches if m.is_file()]

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
        return platform.system().lower()

    @property
    def os_version(self) -> str:
        return f"{platform.system()} {platform.release()}"

    # --- Private helpers ---

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path(self._working_dir) / p

    def _grep_rg(self, pattern: str, path: str, opts: GrepOptions) -> str:
        cmd = ["rg", "--no-heading", "--line-number"]
        if opts.case_insensitive:
            cmd.append("-i")
        if opts.glob_filter:
            cmd.extend(["--glob", opts.glob_filter])
        cmd.extend(["-m", str(opts.max_results), pattern, path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        return result.stdout

    def _grep_python(self, pattern: str, search_path: Path, opts: GrepOptions) -> str:
        flags = re.IGNORECASE if opts.case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex: {e}"

        results: list[str] = []
        paths = [search_path] if search_path.is_file() else sorted(search_path.rglob("*"))

        for fpath in paths:
            if not fpath.is_file():
                continue
            if opts.glob_filter and not fnmatch.fnmatch(fpath.name, opts.glob_filter):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{fpath}:{i}:{line}")
                    if len(results) >= opts.max_results:
                        return "\n".join(results)
        return "\n".join(results)
