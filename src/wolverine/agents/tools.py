"""Custom tools for the Wolverine healing agent."""
from __future__ import annotations

import json
from typing import Any

from agent_loop.tools.registry import ToolDefinition


QUERY_ISSUE = ToolDefinition(
    name="query_issue",
    description="Retrieve full issue details including root cause analysis and related signals.",
    parameters={
        "type": "object",
        "properties": {
            "issue_id": {"type": "string", "description": "The issue ID to query"},
        },
        "required": ["issue_id"],
    },
)

RUN_TESTS = ToolDefinition(
    name="run_tests",
    description="Execute the project test suite and return results.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Test command to run (optional)"},
            "path": {"type": "string", "description": "Specific test file/directory (optional)"},
        },
    },
)


def make_query_issue_executor(issue_data: dict[str, Any]) -> Any:
    """Create an executor that returns issue data as JSON.

    The returned callable has the agent_loop executor signature:
    ``(arguments: dict, env: Any) -> str``
    """
    def executor(arguments: dict[str, Any], env: Any) -> str:
        return json.dumps(issue_data)
    return executor


def make_run_tests_executor(test_command: str = "echo 'no tests configured'") -> Any:
    """Create an executor that runs the test command via the execution environment.

    The returned callable has the agent_loop executor signature:
    ``(arguments: dict, env: Any) -> str``
    """
    def executor(arguments: dict[str, Any], env: Any) -> str:
        cmd = arguments.get("command", test_command)
        timeout = int(arguments.get("timeout_ms", 60_000))
        result = env.exec_command(cmd, timeout_ms=timeout)
        return f"Exit code: {result.exit_code}\nOutput:\n{result.stdout}\n{result.stderr}"
    return executor
