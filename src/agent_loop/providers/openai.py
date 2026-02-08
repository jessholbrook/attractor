"""OpenAI provider profile: codex-rs-aligned tools and prompts."""

from __future__ import annotations

import re
from typing import Any

from agent_loop.environment.types import ExecutionEnvironment
from agent_loop.prompts.builder import build_system_prompt
from agent_loop.tools.core import register_core_tools
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry

# codex-rs-aligned base instructions
_BASE_INSTRUCTIONS = """\
You are a coding assistant powered by OpenAI. You help users with software engineering tasks \
including writing code, debugging, refactoring, and explaining code.

## Tool Usage Guidelines

- Use apply_patch for all file modifications (create, update, delete)
- Use write_file only for creating new files without patch overhead
- Read files before editing to understand existing code
- Use grep and glob to explore the codebase
- Keep changes minimal and focused
- The apply_patch tool uses v4a format with context-based hunks
"""

# --- apply_patch v4a format ---

APPLY_PATCH_DEFINITION = ToolDefinition(
    name="apply_patch",
    description=(
        "Apply code changes using the patch format. "
        "Supports creating, deleting, and modifying files in a single operation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "The patch content in v4a format"},
        },
        "required": ["patch"],
    },
)


def apply_patch_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """Execute a v4a-format patch against the execution environment."""
    patch_text = arguments["patch"]
    return _apply_v4a_patch(patch_text, env)


def _apply_v4a_patch(patch_text: str, env: ExecutionEnvironment) -> str:
    """Parse and apply a v4a format patch.

    Supports:
    - *** Add File: <path> — create new file
    - *** Delete File: <path> — remove file
    - *** Update File: <path> — modify file with context hunks
    - *** Move to: <path> — rename during update
    """
    lines = patch_text.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("Patch must start with '*** Begin Patch'")

    results: list[str] = []
    i = 1  # skip "*** Begin Patch"

    while i < len(lines):
        line = lines[i].strip()

        if line == "*** End Patch":
            break

        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: "):]
            i += 1
            content_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("***"):
                raw = lines[i]
                if raw.startswith("+"):
                    content_lines.append(raw[1:])
                i += 1
            env.write_file(path, "\n".join(content_lines) + ("\n" if content_lines else ""))
            results.append(f"Created {path}")

        elif line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: "):]
            # Write empty to signal deletion (env may not support delete)
            env.write_file(path, "")
            results.append(f"Deleted {path}")
            i += 1

        elif line.startswith("*** Update File: "):
            path = line[len("*** Update File: "):]
            i += 1

            # Check for Move to:
            new_path = None
            if i < len(lines) and lines[i].strip().startswith("*** Move to: "):
                new_path = lines[i].strip()[len("*** Move to: "):]
                i += 1

            # Read existing file
            content = env.read_file(path)
            file_lines = content.splitlines()

            # Parse and apply hunks
            while i < len(lines) and lines[i].startswith("@@ "):
                context_hint = lines[i][3:].strip()
                i += 1
                i, file_lines = _apply_hunk(lines, i, file_lines, context_hint)

            target_path = new_path or path
            env.write_file(target_path, "\n".join(file_lines) + "\n")
            if new_path:
                results.append(f"Updated and moved {path} -> {new_path}")
            else:
                results.append(f"Updated {path}")

        else:
            i += 1

    return "\n".join(results) if results else "No operations performed."


def _apply_hunk(
    patch_lines: list[str],
    start: int,
    file_lines: list[str],
    context_hint: str,
) -> tuple[int, list[str]]:
    """Apply a single hunk to file_lines. Returns (next_index, modified_lines)."""
    # Collect hunk operations
    context: list[str] = []
    removes: list[str] = []
    adds: list[str] = []
    ops: list[tuple[str, str]] = []  # (type, content)

    i = start
    while i < len(patch_lines):
        line = patch_lines[i]
        if line.startswith("@@ ") or line.startswith("***"):
            break
        if line.startswith(" "):
            ops.append(("context", line[1:]))
        elif line.startswith("-"):
            ops.append(("remove", line[1:]))
        elif line.startswith("+"):
            ops.append(("add", line[1:]))
        else:
            break
        i += 1

    # Find the hunk location in file_lines
    # Build the expected sequence of context + remove lines
    expected: list[str] = [content for op, content in ops if op in ("context", "remove")]

    match_pos = _find_match(file_lines, expected, context_hint)

    # Build replacement: context lines + add lines (removing the remove lines)
    replacement: list[str] = []
    for op, content in ops:
        if op == "context":
            replacement.append(content)
        elif op == "add":
            replacement.append(content)
        # "remove" lines are omitted

    new_lines = file_lines[:match_pos] + replacement + file_lines[match_pos + len(expected):]
    return i, new_lines


def _find_match(file_lines: list[str], expected: list[str], context_hint: str) -> int:
    """Find where the expected lines match in the file.

    Tries exact match first, then fuzzy (whitespace-normalized).
    """
    if not expected:
        # No context/remove lines — use hint to find position
        for idx, line in enumerate(file_lines):
            if context_hint and context_hint in line:
                return idx
        return len(file_lines)

    # Exact match
    for start in range(len(file_lines) - len(expected) + 1):
        if file_lines[start:start + len(expected)] == expected:
            return start

    # Fuzzy match (strip whitespace)
    stripped_expected = [s.strip() for s in expected]
    for start in range(len(file_lines) - len(expected) + 1):
        candidate = [s.strip() for s in file_lines[start:start + len(expected)]]
        if candidate == stripped_expected:
            return start

    raise ValueError(
        f"Could not find matching location for hunk with context hint: '{context_hint}'. "
        f"Expected lines: {expected[:3]}..."
    )


class OpenAIProfile:
    """codex-rs-aligned provider profile.

    Uses apply_patch (v4a format) as the native editing tool.
    Default command timeout: 10s per codex-rs convention.
    """

    def __init__(
        self,
        model: str = "o4-mini",
        registry: ToolRegistry | None = None,
    ) -> None:
        self._model = model
        self._registry = registry or ToolRegistry()
        if registry is None:
            # Register core tools, excluding edit_file (replaced by apply_patch)
            register_core_tools(self._registry, exclude={"edit_file"})
            self._registry.register(RegisteredTool(
                definition=APPLY_PATCH_DEFINITION,
                executor=apply_patch_executor,
            ))

    @property
    def id(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    def build_system_prompt(
        self, environment: Any, project_docs: list[str] | None = None
    ) -> str:
        return build_system_prompt(
            base_instructions=_BASE_INSTRUCTIONS,
            environment=environment,
            tool_definitions=self._registry.definitions(),
            model=self._model,
            provider_id="openai",
        )

    def tools(self) -> list[ToolDefinition]:
        return self._registry.definitions()

    def provider_options(self) -> dict[str, Any] | None:
        return None

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_parallel_tool_calls(self) -> bool:
        return True

    @property
    def context_window_size(self) -> int:
        return 128_000
