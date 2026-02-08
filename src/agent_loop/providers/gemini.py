"""Gemini provider profile: gemini-cli-aligned tools and prompts."""

from __future__ import annotations

from typing import Any

from agent_loop.environment.types import ExecutionEnvironment
from agent_loop.prompts.builder import build_system_prompt
from agent_loop.tools.core import register_core_tools
from agent_loop.tools.registry import RegisteredTool, ToolDefinition, ToolRegistry

# gemini-cli-aligned base instructions
_BASE_INSTRUCTIONS = """\
You are a coding assistant powered by Gemini. You help users with software engineering tasks \
including writing code, debugging, refactoring, and explaining code.

## Tool Usage Guidelines

- Read files before editing to understand existing code
- Use edit_file for targeted changes, write_file for full rewrites
- Use list_dir to explore directory structure
- Use grep and glob to search the codebase
- Use shell for running commands, tests, and builds
- Keep changes minimal and focused on the task at hand
"""

# --- Gemini-specific tools ---

LIST_DIR_DEFINITION = ToolDefinition(
    name="list_dir",
    description="List the contents of a directory.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
            "depth": {"type": "integer", "description": "Depth of listing (default: 1)"},
        },
        "required": ["path"],
    },
)


def list_dir_executor(arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
    """List directory contents."""
    entries = env.list_directory(
        path=arguments["path"],
        depth=arguments.get("depth", 1),
    )
    if not entries:
        return "Empty directory."
    lines = []
    for entry in entries:
        suffix = "/" if entry.is_dir else ""
        size_str = f" ({entry.size} bytes)" if entry.size is not None else ""
        lines.append(f"{entry.name}{suffix}{size_str}")
    return "\n".join(lines)


class GeminiProfile:
    """gemini-cli-aligned provider profile.

    Includes list_dir and optionally web_search/web_fetch tools.
    Default command timeout: 10s per gemini-cli convention.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        registry: ToolRegistry | None = None,
    ) -> None:
        self._model = model
        self._registry = registry or ToolRegistry()
        if registry is None:
            register_core_tools(self._registry)
            self._registry.register(RegisteredTool(
                definition=LIST_DIR_DEFINITION,
                executor=list_dir_executor,
            ))

    @property
    def id(self) -> str:
        return "gemini"

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
            provider_id="gemini",
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
        return 1_000_000
