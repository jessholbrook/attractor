"""Shared core tool definitions and executor factories.

All provider profiles share these base tools. The parameter schemas and
executor implementations delegate to the ExecutionEnvironment interface.
"""

from __future__ import annotations

from typing import Any

from agent_loop.environment.types import ExecutionEnvironment, GrepOptions
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry


# --- Tool Definitions (JSON Schema) ---

READ_FILE_DEFINITION = ToolDefinition(
    name="read_file",
    description="Read a file from the filesystem. Returns line-numbered content.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "offset": {"type": "integer", "description": "1-based line number to start reading from"},
            "limit": {"type": "integer", "description": "Max lines to read (default: 2000)"},
        },
        "required": ["file_path"],
    },
)

WRITE_FILE_DEFINITION = ToolDefinition(
    name="write_file",
    description="Write content to a file. Creates the file and parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path"},
            "content": {"type": "string", "description": "The full file content"},
        },
        "required": ["file_path", "content"],
    },
)

EDIT_FILE_DEFINITION = ToolDefinition(
    name="edit_file",
    description="Replace an exact string occurrence in a file.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "old_string": {"type": "string", "description": "Exact text to find"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default: false)"},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
)

SHELL_DEFINITION = ToolDefinition(
    name="shell",
    description="Execute a shell command. Returns stdout, stderr, and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
            "timeout_ms": {"type": "integer", "description": "Override default timeout in milliseconds"},
            "description": {"type": "string", "description": "Human-readable description"},
        },
        "required": ["command"],
    },
)

GREP_DEFINITION = ToolDefinition(
    name="grep",
    description="Search file contents using regex patterns.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "path": {"type": "string", "description": "Directory or file to search"},
            "glob_filter": {"type": "string", "description": "File pattern filter (e.g., '*.py')"},
            "case_insensitive": {"type": "boolean", "description": "Case insensitive search"},
            "max_results": {"type": "integer", "description": "Maximum results (default: 100)"},
        },
        "required": ["pattern"],
    },
)

GLOB_DEFINITION = ToolDefinition(
    name="glob",
    description="Find files matching a glob pattern.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.ts')"},
            "path": {"type": "string", "description": "Base directory"},
        },
        "required": ["pattern"],
    },
)

# All core definitions by name
CORE_TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "read_file": READ_FILE_DEFINITION,
    "write_file": WRITE_FILE_DEFINITION,
    "edit_file": EDIT_FILE_DEFINITION,
    "shell": SHELL_DEFINITION,
    "grep": GREP_DEFINITION,
    "glob": GLOB_DEFINITION,
}


# --- Tool Executors ---


def read_file_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Read a file and return line-numbered content."""
    content = env.read_file(
        path=arguments["file_path"],
        offset=arguments.get("offset"),
        limit=arguments.get("limit", 2000),
    )
    lines = content.splitlines()
    start = arguments.get("offset", 1) or 1
    numbered = [f"{start + i:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def write_file_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Write content to a file."""
    env.write_file(arguments["file_path"], arguments["content"])
    byte_count = len(arguments["content"].encode("utf-8"))
    return f"Wrote {byte_count} bytes to {arguments['file_path']}"


def edit_file_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Replace exact string in a file."""
    path = arguments["file_path"]
    old = arguments["old_string"]
    new = arguments["new_string"]
    replace_all = arguments.get("replace_all", False)

    content = env.read_file(path)
    count = content.count(old)

    if count == 0:
        raise ValueError(f"old_string not found in {path}")
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string found {count} times in {path}. "
            "Provide more context to make it unique, or set replace_all=true."
        )

    if replace_all:
        new_content = content.replace(old, new)
    else:
        new_content = content.replace(old, new, 1)

    env.write_file(path, new_content)
    replacements = count if replace_all else 1
    return f"Made {replacements} replacement(s) in {path}"


def shell_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Execute a shell command."""
    result = env.exec_command(
        command=arguments["command"],
        timeout_ms=arguments.get("timeout_ms", 10_000),
    )
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"STDERR: {result.stderr}")
    parts.append(f"Exit code: {result.exit_code}")
    if result.timed_out:
        parts.append(f"(timed out after {result.duration_ms}ms)")
    return "\n".join(parts)


def grep_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Search file contents by pattern."""
    options = GrepOptions(
        case_insensitive=arguments.get("case_insensitive", False),
        glob_filter=arguments.get("glob_filter"),
        max_results=arguments.get("max_results", 100),
    )
    return env.grep(
        pattern=arguments["pattern"],
        path=arguments.get("path", "."),
        options=options,
    )


def glob_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Find files by glob pattern."""
    matches = env.glob(
        pattern=arguments["pattern"],
        path=arguments.get("path", "."),
    )
    return "\n".join(matches) if matches else "No files found."


# Executor map
CORE_TOOL_EXECUTORS: dict[str, Any] = {
    "read_file": read_file_executor,
    "write_file": write_file_executor,
    "edit_file": edit_file_executor,
    "shell": shell_executor,
    "grep": grep_executor,
    "glob": glob_executor,
}


# --- Registration helper ---


def register_core_tools(
    registry: ToolRegistry,
    exclude: set[str] | None = None,
) -> None:
    """Register all core tools onto a registry.

    Args:
        registry: Target ToolRegistry.
        exclude: Tool names to skip (e.g., {"edit_file"} for OpenAI profile).
    """
    skip = exclude or set()
    for name, definition in CORE_TOOL_DEFINITIONS.items():
        if name in skip:
            continue
        registry.register(RegisteredTool(
            definition=definition,
            executor=CORE_TOOL_EXECUTORS[name],
        ))
