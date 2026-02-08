"""Provider profile protocol and stub implementation."""

from __future__ import annotations

from typing import Any, Protocol

from agent_loop.tools.registry import ToolDefinition, ToolRegistry


class ProviderProfile(Protocol):
    """Protocol for provider-aligned tool and prompt profiles.

    Each model family works best with its native agent's tools and system prompts.
    A profile bundles the tools, system prompt, and configuration for a specific
    provider (OpenAI, Anthropic, Gemini).
    """

    @property
    def id(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def tool_registry(self) -> ToolRegistry: ...

    def build_system_prompt(
        self, environment: Any, project_docs: list[str] | None = None
    ) -> str: ...

    def tools(self) -> list[ToolDefinition]: ...

    def provider_options(self) -> dict[str, Any] | None: ...

    # Capability flags
    @property
    def supports_reasoning(self) -> bool: ...

    @property
    def supports_streaming(self) -> bool: ...

    @property
    def supports_parallel_tool_calls(self) -> bool: ...

    @property
    def context_window_size(self) -> int: ...


class StubProfile:
    """Minimal ProviderProfile implementation for tests.

    Provides configurable defaults and satisfies the ProviderProfile protocol.
    """

    def __init__(
        self,
        *,
        profile_id: str = "stub",
        model: str = "stub-model",
        registry: ToolRegistry | None = None,
        supports_reasoning: bool = False,
        supports_streaming: bool = False,
        supports_parallel: bool = True,
        context_window: int = 128_000,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self._id = profile_id
        self._model = model
        self._registry = registry or ToolRegistry()
        self._supports_reasoning = supports_reasoning
        self._supports_streaming = supports_streaming
        self._supports_parallel = supports_parallel
        self._context_window = context_window
        self._system_prompt = system_prompt

    @property
    def id(self) -> str:
        return self._id

    @property
    def model(self) -> str:
        return self._model

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    def build_system_prompt(
        self, environment: Any, project_docs: list[str] | None = None
    ) -> str:
        return self._system_prompt

    def tools(self) -> list[ToolDefinition]:
        return self._registry.definitions()

    def provider_options(self) -> dict[str, Any] | None:
        return None

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def supports_streaming(self) -> bool:
        return self._supports_streaming

    @property
    def supports_parallel_tool_calls(self) -> bool:
        return self._supports_parallel

    @property
    def context_window_size(self) -> int:
        return self._context_window
