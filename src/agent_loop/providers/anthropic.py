"""Anthropic provider profile: Claude Code-aligned tools and prompts."""

from __future__ import annotations

from typing import Any

from agent_loop.prompts.builder import build_system_prompt
from agent_loop.tools.core import register_core_tools
from agent_loop.tools.registry import ToolDefinition, ToolRegistry

# Claude Code-aligned base instructions
_BASE_INSTRUCTIONS = """\
You are a coding assistant powered by Claude. You help users with software engineering tasks \
including writing code, debugging, refactoring, and explaining code.

## Tool Usage Guidelines

- Read files before editing them to understand existing code
- Prefer editing existing files over creating new ones
- When using edit_file, the old_string must be unique in the file
- Use shell for running commands, tests, and builds
- Use grep and glob to explore the codebase before making changes
- Keep changes minimal and focused on the task at hand
"""


class AnthropicProfile:
    """Claude Code-aligned provider profile.

    Uses edit_file (old_string/new_string) as the native editing format.
    Default command timeout: 120s per Claude Code convention.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        registry: ToolRegistry | None = None,
    ) -> None:
        self._model = model
        self._registry = registry or ToolRegistry()
        if registry is None:
            register_core_tools(self._registry)

    @property
    def id(self) -> str:
        return "anthropic"

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
            provider_id="anthropic",
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
        return 200_000
