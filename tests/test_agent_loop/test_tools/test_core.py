"""Tests for shared core tool definitions and executors."""

import pytest

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.environment.types import ExecResult
from agent_loop.tools.core import (
    CORE_TOOL_DEFINITIONS,
    CORE_TOOL_EXECUTORS,
    edit_file_executor,
    glob_executor,
    grep_executor,
    read_file_executor,
    register_core_tools,
    shell_executor,
    write_file_executor,
)
from agent_loop.tools.registry import ToolRegistry


# --- Definitions ---


class TestCoreToolDefinitions:
    def test_has_six_core_tools(self):
        assert len(CORE_TOOL_DEFINITIONS) == 6

    def test_read_file_definition(self):
        d = CORE_TOOL_DEFINITIONS["read_file"]
        assert d.name == "read_file"
        assert "file_path" in d.parameters["properties"]
        assert "file_path" in d.parameters["required"]

    def test_write_file_definition(self):
        d = CORE_TOOL_DEFINITIONS["write_file"]
        assert "content" in d.parameters["required"]

    def test_edit_file_definition(self):
        d = CORE_TOOL_DEFINITIONS["edit_file"]
        assert "old_string" in d.parameters["required"]
        assert "new_string" in d.parameters["required"]

    def test_shell_definition(self):
        d = CORE_TOOL_DEFINITIONS["shell"]
        assert "command" in d.parameters["required"]

    def test_grep_definition(self):
        d = CORE_TOOL_DEFINITIONS["grep"]
        assert "pattern" in d.parameters["required"]

    def test_glob_definition(self):
        d = CORE_TOOL_DEFINITIONS["glob"]
        assert "pattern" in d.parameters["required"]


# --- Executors ---


class TestReadFileExecutor:
    def test_returns_line_numbered_content(self):
        env = StubExecutionEnvironment(files={"/app/main.py": "import os\nprint('hi')\n"})
        result = read_file_executor({"file_path": "/app/main.py"}, env)
        assert "1 | import os" in result
        assert "2 | print('hi')" in result

    def test_raises_on_missing_file(self):
        env = StubExecutionEnvironment()
        with pytest.raises(FileNotFoundError):
            read_file_executor({"file_path": "/missing"}, env)


class TestWriteFileExecutor:
    def test_writes_and_reports_bytes(self):
        env = StubExecutionEnvironment()
        result = write_file_executor({"file_path": "/new.txt", "content": "hello"}, env)
        assert "5 bytes" in result
        assert env.read_file("/new.txt") == "hello"


class TestEditFileExecutor:
    def test_replaces_exact_match(self):
        env = StubExecutionEnvironment(files={"/f": "hello world"})
        result = edit_file_executor({"file_path": "/f", "old_string": "hello", "new_string": "goodbye"}, env)
        assert "1 replacement" in result
        assert env.read_file("/f") == "goodbye world"

    def test_raises_when_not_found(self):
        env = StubExecutionEnvironment(files={"/f": "hello"})
        with pytest.raises(ValueError, match="not found"):
            edit_file_executor({"file_path": "/f", "old_string": "xyz", "new_string": "abc"}, env)

    def test_raises_when_not_unique(self):
        env = StubExecutionEnvironment(files={"/f": "aa bb aa"})
        with pytest.raises(ValueError, match="2 times"):
            edit_file_executor({"file_path": "/f", "old_string": "aa", "new_string": "cc"}, env)

    def test_replace_all(self):
        env = StubExecutionEnvironment(files={"/f": "aa bb aa"})
        result = edit_file_executor(
            {"file_path": "/f", "old_string": "aa", "new_string": "cc", "replace_all": True}, env
        )
        assert "2 replacement" in result
        assert env.read_file("/f") == "cc bb cc"


class TestShellExecutor:
    def test_formats_exec_result(self):
        env = StubExecutionEnvironment(exec_results=[ExecResult(stdout="output\n", exit_code=0)])
        result = shell_executor({"command": "echo hi"}, env)
        assert "output" in result
        assert "Exit code: 0" in result

    def test_includes_stderr(self):
        env = StubExecutionEnvironment(exec_results=[ExecResult(stderr="warn", exit_code=1)])
        result = shell_executor({"command": "bad"}, env)
        assert "STDERR: warn" in result

    def test_reports_timeout(self):
        env = StubExecutionEnvironment(exec_results=[
            ExecResult(stdout="partial", timed_out=True, exit_code=-1, duration_ms=5000)
        ])
        result = shell_executor({"command": "sleep 99"}, env)
        assert "timed out" in result


class TestGrepExecutor:
    def test_delegates_to_env(self):
        env = StubExecutionEnvironment()
        result = grep_executor({"pattern": "foo"}, env)
        assert isinstance(result, str)


class TestGlobExecutor:
    def test_returns_matched_files(self):
        env = StubExecutionEnvironment(files={"/app/a.py": "", "/app/b.py": ""})
        result = glob_executor({"pattern": "*.py", "path": "/app"}, env)
        assert "/app/a.py" in result

    def test_returns_no_files_message(self):
        env = StubExecutionEnvironment()
        result = glob_executor({"pattern": "*.rs"}, env)
        assert "No files found" in result


# --- Registration ---


class TestRegisterCoreTools:
    def test_registers_all_six(self):
        reg = ToolRegistry()
        register_core_tools(reg)
        assert len(reg.names()) == 6

    def test_exclude_removes_tools(self):
        reg = ToolRegistry()
        register_core_tools(reg, exclude={"edit_file"})
        assert "edit_file" not in reg.names()
        assert len(reg.names()) == 5

    def test_registered_tools_are_callable(self):
        reg = ToolRegistry()
        register_core_tools(reg)
        for name in reg.names():
            tool = reg.get(name)
            assert callable(tool.executor)
