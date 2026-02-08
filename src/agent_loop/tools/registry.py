"""Tool registry: definitions, registration, and lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    """Schema definition for a tool exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema, root type "object"


@dataclass(frozen=True)
class RegisteredTool:
    """A tool definition paired with its executor function.

    The executor signature is (arguments: dict, execution_env: Any) -> str.
    """

    definition: ToolDefinition
    executor: Callable[[dict[str, Any], Any], str]


class ToolRegistry:
    """Registry of tools available to a provider profile.

    Supports register, unregister, lookup, and listing.
    Latest-wins on name collision. Insertion-order stable.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register a tool. Overwrites any existing tool with the same name."""
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions in registration order."""
        return [t.definition for t in self._tools.values()]

    def names(self) -> list[str]:
        """Return all registered tool names in registration order."""
        return list(self._tools.keys())
