"""Tests for Wolverine custom agent tools."""
from __future__ import annotations

import json

import pytest

from agent_loop.environment.stub import StubExecutionEnvironment
from agent_loop.environment.types import ExecResult

from wolverine.agents.tools import (
    QUERY_ISSUE,
    RUN_TESTS,
    make_query_issue_executor,
    make_run_tests_executor,
)


# ---------------------------------------------------------------------------
# Tool definition tests
# ---------------------------------------------------------------------------

class TestToolDefinitions:

    def test_query_issue_name(self):
        assert QUERY_ISSUE.name == "query_issue"

    def test_query_issue_has_description(self):
        assert QUERY_ISSUE.description
        assert len(QUERY_ISSUE.description) > 10

    def test_query_issue_parameters_require_issue_id(self):
        assert "issue_id" in QUERY_ISSUE.parameters["properties"]
        assert "issue_id" in QUERY_ISSUE.parameters["required"]

    def test_run_tests_name(self):
        assert RUN_TESTS.name == "run_tests"

    def test_run_tests_has_description(self):
        assert RUN_TESTS.description
        assert len(RUN_TESTS.description) > 10

    def test_run_tests_parameters_has_command(self):
        assert "command" in RUN_TESTS.parameters["properties"]

    def test_run_tests_parameters_has_path(self):
        assert "path" in RUN_TESTS.parameters["properties"]

    def test_run_tests_no_required_fields(self):
        # Both command and path are optional
        assert "required" not in RUN_TESTS.parameters


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------

class TestQueryIssueExecutor:

    def test_returns_json(self):
        data = {"id": "ISS-1", "title": "Bug", "severity": "high"}
        executor = make_query_issue_executor(data)
        env = StubExecutionEnvironment()
        result = executor({"issue_id": "ISS-1"}, env)
        parsed = json.loads(result)
        assert parsed["id"] == "ISS-1"
        assert parsed["title"] == "Bug"

    def test_returns_all_fields(self):
        data = {"id": "ISS-2", "title": "Error", "root_cause": "null pointer", "affected_files": ["/a.py"]}
        executor = make_query_issue_executor(data)
        env = StubExecutionEnvironment()
        result = json.loads(executor({"issue_id": "ISS-2"}, env))
        assert result["root_cause"] == "null pointer"
        assert result["affected_files"] == ["/a.py"]

    def test_ignores_arguments(self):
        """The executor always returns the same data regardless of arguments."""
        data = {"id": "ISS-1"}
        executor = make_query_issue_executor(data)
        env = StubExecutionEnvironment()
        result_a = executor({"issue_id": "ISS-1"}, env)
        result_b = executor({"issue_id": "ISS-99"}, env)
        assert result_a == result_b


class TestRunTestsExecutor:

    def test_calls_exec_command(self):
        env = StubExecutionEnvironment(exec_results=[
            ExecResult(stdout="PASSED", exit_code=0),
        ])
        executor = make_run_tests_executor("pytest tests/")
        result = executor({}, env)
        assert "Exit code: 0" in result
        assert "PASSED" in result
        assert env.exec_calls == ["pytest tests/"]

    def test_uses_argument_command_over_default(self):
        env = StubExecutionEnvironment(exec_results=[
            ExecResult(stdout="ok", exit_code=0),
        ])
        executor = make_run_tests_executor("pytest")
        result = executor({"command": "npm test"}, env)
        assert env.exec_calls == ["npm test"]

    def test_includes_stderr_in_output(self):
        env = StubExecutionEnvironment(exec_results=[
            ExecResult(stdout="", stderr="ERROR: module not found", exit_code=1),
        ])
        executor = make_run_tests_executor()
        result = executor({}, env)
        assert "ERROR: module not found" in result
        assert "Exit code: 1" in result

    def test_default_command_when_no_argument(self):
        env = StubExecutionEnvironment(exec_results=[ExecResult()])
        executor = make_run_tests_executor("echo 'no tests configured'")
        executor({}, env)
        assert env.exec_calls == ["echo 'no tests configured'"]
