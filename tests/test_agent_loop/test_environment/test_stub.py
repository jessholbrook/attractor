"""Tests for the stub execution environment."""

import pytest

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.environment.types import ExecResult, ExecutionEnvironment


class TestStubExecutionEnvironmentProtocol:
    def test_satisfies_protocol(self):
        env: ExecutionEnvironment = StubExecutionEnvironment()
        assert env.working_directory == "/stub/workspace"


class TestStubExecutionEnvironmentFiles:
    def test_read_file_returns_configured_content(self):
        env = StubExecutionEnvironment(files={"/app/main.py": "print('hello')"})
        assert env.read_file("/app/main.py") == "print('hello')"

    def test_read_file_raises_for_missing_file(self):
        env = StubExecutionEnvironment()
        with pytest.raises(FileNotFoundError):
            env.read_file("/missing.txt")

    def test_read_file_with_offset_and_limit(self):
        env = StubExecutionEnvironment(files={"/f": "line1\nline2\nline3\nline4\n"})
        result = env.read_file("/f", offset=2, limit=2)
        assert result == "line2\nline3\n"

    def test_write_file_stores_content(self):
        env = StubExecutionEnvironment()
        env.write_file("/new.txt", "content")
        assert env.read_file("/new.txt") == "content"

    def test_file_exists_true(self):
        env = StubExecutionEnvironment(files={"/a": "data"})
        assert env.file_exists("/a") is True

    def test_file_exists_false(self):
        env = StubExecutionEnvironment()
        assert env.file_exists("/nope") is False


class TestStubExecutionEnvironmentExec:
    def test_returns_configured_results(self):
        r = ExecResult(stdout="output", exit_code=0)
        env = StubExecutionEnvironment(exec_results=[r])
        assert env.exec_command("echo hi") == r

    def test_cycles_last_result(self):
        r = ExecResult(stdout="always")
        env = StubExecutionEnvironment(exec_results=[r])
        env.exec_command("cmd1")
        result = env.exec_command("cmd2")
        assert result.stdout == "always"

    def test_tracks_exec_calls(self):
        env = StubExecutionEnvironment()
        env.exec_command("ls")
        env.exec_command("pwd")
        assert env.exec_calls == ["ls", "pwd"]


class TestStubExecutionEnvironmentMetadata:
    def test_working_directory(self):
        env = StubExecutionEnvironment(working_dir="/custom")
        assert env.working_directory == "/custom"

    def test_platform(self):
        env = StubExecutionEnvironment(plat="linux")
        assert env.platform == "linux"

    def test_os_version(self):
        env = StubExecutionEnvironment(os_ver="Linux 6.1.0")
        assert env.os_version == "Linux 6.1.0"


class TestStubExecutionEnvironmentLifecycle:
    def test_initialize_is_noop(self):
        env = StubExecutionEnvironment()
        env.initialize()  # should not raise

    def test_cleanup_is_noop(self):
        env = StubExecutionEnvironment()
        env.cleanup()  # should not raise
