"""Tests for the local execution environment."""

import os
import platform as platform_mod

import pytest

from agent_loop.environment.local import (
    EnvVarPolicy,
    LocalExecutionEnvironment,
    _filter_env,
    _is_sensitive,
)
from agent_loop.environment.types import ExecutionEnvironment


# --- Protocol satisfaction ---


class TestLocalExecutionEnvironmentProtocol:
    def test_satisfies_protocol(self):
        env: ExecutionEnvironment = LocalExecutionEnvironment()
        assert env.working_directory


# --- File operations ---


class TestLocalFileOps:
    def test_read_file_returns_content(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        assert env.read_file(str(f)) == "hello world"

    def test_read_file_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "lines.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.read_file(str(f), offset=2, limit=2)
        assert result == "b\nc\n"

    def test_read_file_nonexistent_raises(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            env.read_file(str(tmp_path / "nope.txt"))

    def test_write_file_creates_file(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        target = str(tmp_path / "new.txt")
        env.write_file(target, "content")
        assert (tmp_path / "new.txt").read_text() == "content"

    def test_write_file_creates_parent_dirs(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        target = str(tmp_path / "sub" / "deep" / "file.txt")
        env.write_file(target, "nested")
        assert (tmp_path / "sub" / "deep" / "file.txt").read_text() == "nested"

    def test_write_file_overwrites_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        env.write_file(str(f), "new")
        assert f.read_text() == "new"

    def test_file_exists_true(self, tmp_path):
        (tmp_path / "yes.txt").write_text("x")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        assert env.file_exists(str(tmp_path / "yes.txt")) is True

    def test_file_exists_false(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        assert env.file_exists(str(tmp_path / "no.txt")) is False

    def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "subdir").mkdir()
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        entries = env.list_directory(str(tmp_path))
        names = {e.name for e in entries}
        assert "a.txt" in names
        assert "subdir" in names
        file_entry = next(e for e in entries if e.name == "a.txt")
        assert file_entry.is_dir is False
        assert file_entry.size is not None
        dir_entry = next(e for e in entries if e.name == "subdir")
        assert dir_entry.is_dir is True


# --- Command execution ---


class TestLocalExecCommand:
    def test_exec_simple_command(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("echo hello")
        assert result.stdout.strip() == "hello"
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_exec_captures_stderr(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("echo err >&2")
        assert "err" in result.stderr

    def test_exec_returns_exit_code(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("exit 42")
        assert result.exit_code == 42

    def test_exec_timeout(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("sleep 10", timeout_ms=200)
        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stdout.lower()

    def test_exec_duration_tracked(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("echo fast")
        assert result.duration_ms >= 0

    def test_exec_custom_working_dir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("pwd", working_dir=str(sub))
        assert result.stdout.strip() == str(sub)

    def test_exec_custom_env_vars(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.exec_command("echo $MY_VAR", env_vars={"MY_VAR": "hello"})
        assert result.stdout.strip() == "hello"


# --- Environment variable filtering ---


class TestEnvVarFiltering:
    def test_is_sensitive_api_key(self):
        assert _is_sensitive("OPENAI_API_KEY") is True
        assert _is_sensitive("openai_api_key") is True

    def test_is_sensitive_secret(self):
        assert _is_sensitive("DB_SECRET") is True

    def test_is_sensitive_token(self):
        assert _is_sensitive("GITHUB_TOKEN") is True

    def test_is_not_sensitive(self):
        assert _is_sensitive("PATH") is False
        assert _is_sensitive("HOME") is False
        assert _is_sensitive("EDITOR") is False

    def test_inherit_core_filters_sensitive(self):
        os.environ["TEST_TEMP_API_KEY"] = "secret"
        try:
            env = _filter_env(EnvVarPolicy.INHERIT_CORE)
            assert "TEST_TEMP_API_KEY" not in env
            assert "PATH" in env
        finally:
            del os.environ["TEST_TEMP_API_KEY"]

    def test_inherit_all_passes_everything(self):
        os.environ["TEST_TEMP_API_KEY"] = "secret"
        try:
            env = _filter_env(EnvVarPolicy.INHERIT_ALL)
            assert "TEST_TEMP_API_KEY" in env
        finally:
            del os.environ["TEST_TEMP_API_KEY"]

    def test_inherit_none_only_core(self):
        env = _filter_env(EnvVarPolicy.INHERIT_NONE)
        assert "PATH" in env
        # Random env vars should not be present
        for key in env:
            assert key in {"PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR",
                           "GOPATH", "CARGO_HOME", "NVM_DIR", "PYTHONPATH", "VIRTUAL_ENV",
                           "PYENV_ROOT", "RBENV_ROOT", "RUSTUP_HOME"}


# --- Metadata ---


class TestLocalMetadata:
    def test_working_directory(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        assert env.working_directory == str(tmp_path)

    def test_working_directory_defaults_to_cwd(self):
        env = LocalExecutionEnvironment()
        assert env.working_directory == os.getcwd()

    def test_platform(self):
        env = LocalExecutionEnvironment()
        assert env.platform == platform_mod.system().lower()

    def test_os_version(self):
        env = LocalExecutionEnvironment()
        assert platform_mod.system() in env.os_version


# --- Search operations ---


class TestLocalGrep:
    def test_grep_finds_pattern(self, tmp_path):
        (tmp_path / "hello.py").write_text("def greet():\n    print('hello')\n")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.grep("greet", str(tmp_path))
        assert "greet" in result

    def test_grep_case_insensitive(self, tmp_path):
        (tmp_path / "test.txt").write_text("Hello World\n")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        from agent_loop.environment.types import GrepOptions
        result = env.grep("hello", str(tmp_path), GrepOptions(case_insensitive=True))
        assert "Hello" in result

    def test_grep_max_results(self, tmp_path):
        (tmp_path / "many.txt").write_text("\n".join(f"match line {i}" for i in range(100)))
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        from agent_loop.environment.types import GrepOptions
        result = env.grep("match", str(tmp_path), GrepOptions(max_results=5))
        assert result.count("\n") <= 5


class TestLocalGlob:
    def test_glob_finds_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.glob("*.py", str(tmp_path))
        assert len(result) == 2
        assert all(r.endswith(".py") for r in result)

    def test_glob_empty_result(self, tmp_path):
        env = LocalExecutionEnvironment(working_dir=str(tmp_path))
        result = env.glob("*.rs", str(tmp_path))
        assert result == []


# --- Lifecycle ---


class TestLocalLifecycle:
    def test_initialize(self):
        env = LocalExecutionEnvironment()
        env.initialize()  # should not raise

    def test_cleanup(self):
        env = LocalExecutionEnvironment()
        env.cleanup()  # should not raise
